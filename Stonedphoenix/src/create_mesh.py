# input for iFieldstone 

import os 
import sys
sys.path.append(os.path.abspath("src"))
import Function_make_mesh 
from Function_make_mesh import Class_Points
from Function_make_mesh import Class_Line
from Function_make_mesh import create_loop
from Function_make_mesh import find_line_index
from numerical_control import IOControls

import numpy as np
import gmsh 

import ufl
import meshio
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, io, nls, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import matplotlib.pyplot as plt
from dolfinx.io import XDMFFile, gmshio
import gmsh 
from ufl import exp, conditional, eq, as_ufl
import scal as sc_f 
import basix.ufl

# dictionary for surface and phase. 
"""
Updated Branch: The mesh is created as a function of the slab geometry: first, the slab geometry is either defined by an input file (To do) 
                or by the class that I introduced that create a slab geometry as a function of lambda function (to do this particular feature). 
                Then, I create oceanic crust, bottom of the slab, and crustal units and store relevant information in the fundamental points array 
                ----
                After that i create the points with gmesh, then the line, physical line {save the dictionary} 
                ----                                         Domain            [Phase]
                After that I create the physical domains: a. Subduction plate  [1. Slab s.s.
                                                                                2. Oceanic plate]
                                                          b. Mantle Wedge       [3. Wedge ]
                                                          c. Overriding         [4. Lithospheric mantle s.s.
                                                                                 5. Crust (upper crust)
                                                                                 6. Lower crust (optional)]
                                                        The domain are the geometrical entities that I use to solve the equations. They are composed of a several 
                                                        phases which represents the material property. 
                                                        So the script for generating the mesh is organise as follow -> 
                                                        ====
                                                        Domain A [Subduction plate]        -> create lines, create loops, create surfaces
                                                        Domain B [Wedge plate]             -> create lines, create loops, create surfaces
                                                        Domain C [Crust -> the useless one]-> create lines, create loops, create surfaces
                                                        ====
                                                        -> Create a class where to store all these data -> and pass to the main code for the resolution
                                                        ---
                                                        I dislike the usage of the class, i prefer to distinguis the material property
                                                        and use ad hoc function within the resolution of the system. I don't know what are the benefits, but looking a few repository 
                                                        it seems a bit chaotic.
                                   
                In this updated version I am not introducing a channel. I have the impression that my way to produce the setup is too rigid, and I think 
                that after the first working example would be worth to simplify the work flow. 
                General rule: If I copy or take ispiration for pre-existing software I cite in the function where do I steal the idea. 
"""

class geom_input():
    def __init__(self,
                 x = np.array([0.,1000e3]),
                 y = np.array([-660e3,0.]),
                 cr=35e3,
                 ocr=6e3,
                 lit_mt=30e3,
                 lc = 0.5,
                 wc = 1.5e3,
                 slab_tk = 130e3, 
                 decoupling = 100e3):
        self.x                 = x               # main grid coordinate
        self.y                 = y   
        self.slab_tk           = slab_tk
        self.cr                = cr              # crust 
        self.ocr               = ocr             # oceanic crust
        self.lit_mt            = lit_mt          # lithosperic mantle  
        self.lc                = lc              # lower crust ratio 
        self.wc                = wc              # weak zone 
        self.lt_d              = (cr+lit_mt)     # total lithosphere thickness
        self.decoupling        = decoupling      # decoupling depth -> i.e. where the weak zone is prolonged 
        self.resolution_normal = wc*2  # To Do
    
    def dimensionless_ginput(self,sc):
        self.x                 /= sc.L               # main grid coordinate
        self.y                 /= sc.L   
        self.cr                /= sc.L              # crust 
        self.ocr               /= sc.L             # oceanic crust
        self.lit_mt            /= sc.L          # lithosperic mantle  
        self.wc                /= sc.L             # weak zone 
        self.lt_d              /= sc.L    # total lithosphere thickness
        self.decoupling        /= sc.L      # decoupling depth -> i.e. where the weak zone is prolonged 
        self.resolution_normal /= sc.L  # To Do
        
        return self 
        
        


def assign_phases(dict_surf, cell_tags,phase):
    """Assigns phase tags to the mesh based on the provided surface tags."""
    for tag, value in dict_surf.items():
        indices = cell_tags.find(value) 
        phase.x.array[indices] = np.full_like(indices,  value , dtype=PETSc.IntType)
    
    return phase 


class Mesh(): 
    def __init__(self) :

        " Class where to store the information of the mesh"

        self.mesh      : dolfinx.mesh.Mesh                     # Mesh 
        self.mesh_Ctag : dolfinx.mesh.MeshTags                 # Mesh cell tag       {i.e., Physical surface, where I define the phase properties}
        self.mesh_Ftag : dolfinx.mesh.MeshTags                 # Mesh cell face tage {i.e., Face tag, where the information of the internal boundary are stored}
        self.Pph       : dolfinx.fem.function.FunctionSpace    # Function space      [mesh + element type + dof]
        self.PT        : dolfinx.fem.function.FunctionSpace    # Function space 
        self.PD        : dolphinx.fem.function.FunctionSpace   # Function space      
        self.V         : dolphinx.fem.function.FunctionSpace   # Vectorial function space 
        self.phase     : dolfinx.fem.function.Function         # Function a Field (solution potential of the function space)
        self.T_i       : dolfinx.fem.function.Function         # Function      

        

       



def create_mesh(mesh, cell_type, prune_z=False):
    """
    mesh, cell_type, remove z 
    
    
    
    
    
    source: Dokken tutorial generating mesh 
    """
    
    # From the tutorials of dolfinx
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data.astype(np.int32)]})
    return out_mesh


dict_surf = {
    'sub_plate'         : 1,
    'oceanic_crust'     : 2,
    'wedge'             : 3,
    'overriding_lm'     : 4,
    'upper_crust'       : 5,
    'lower_crust'       : 6,

}

dict_tag_lines = {
    'Top'               : 1,
    'Right'             : 2,
    'Bottom'            : 3,
    'Left'              : 4,
    'Subduction_top'    : 5,
    'Subduction_bot'    : 6,
    'Oceanic'           : 7,
    'Overriding_mantle' : 9,
    'Channel_decoupling': 10,
    'Crust_overplate'   : 11,
    'LCrust_overplate'  : 12,
    'Full_over'         : 13,
}

def create_domainA():
    
    
    pass


def create_domainB():
    
    pass 


def create_domainC():

    pass 



def create_gmsh(sx,        # subduction x
                sy,        # subdcution y 
                bsx,       # bottom subduction x
                bsy,       # bottom subduction y 
                oc_cx,     # oceanic cx 
                oc_cy,     # oceanic cu
                g_input,
                fp):  # geometry input class 
    
    # -> USE GMSH FUNCTION 
    gmsh.initialize()


    mesh_model = gmsh.model()

    CP = Class_Points()
    mesh_model = CP.update_points(mesh_model,sx,sy,bsx,bsy,oc_cx,oc_cy,g_input,fp)

    LC = Class_Line()
    mesh_model = LC.update_lines(mesh_model,CP, g_input)


    # Function create Physical line
    #-- Create Physical Line
    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}
    
    mesh_model.addPhysicalGroup(1, LC.tag_L_T, tag=dict_tag_lines['Top'])
    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, LC.tag_L_R, tag=dict_tag_lines['Right'])


    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, LC.tag_L_B, tag=dict_tag_lines['Bottom'])

    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, LC.tag_L_L, tag=dict_tag_lines['Left'])

    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, LC.tag_L_sub,   tag=dict_tag_lines['Subduction'])

    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, LC.tag_L_ch,    tag=dict_tag_lines['Channel'])
    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, LC.tag_L_oc,    tag=dict_tag_lines['Oceanic'])
  
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, LC.tag_L_ch_ov, tag=dict_tag_lines['Channel_over']) 

    gmsh.model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, LC.tag_L_ov,    tag=dict_tag_lines['Overriding_mantle'])

    gmsh.model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, [LC.tag_L_ov[0],LC.tag_L_ch_ov[0]] , tag=dict_tag_lines['Full_over'])

    if g_input.cr !=0: 

        mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

        mesh_model.addPhysicalGroup(1,    LC.tag_L_cr, tag=dict_tag_lines['Crust_overplate'])

        if g_input.lc !=0:
            mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}
            mesh_model.addPhysicalGroup(1, LC.tag_L_Lcr,tag=dict_tag_lines['LCrust_overplate'])
   
      
    # Create the line loop [anticlockwise] -> TODO automatic detection orientation of the line, and a more automatic way to select the lines   
    # Incoming plate  
    
    
    
    l_list     = [LC.lines_oc[2,:], LC.lines_B[2,-1], -LC.lines_L[2,0]]
    mesh_model = create_loop(l_list, mesh_model, 10)

    # Oceanic crust  
    l_list     = [LC.lines_S[2,:],   LC.lines_B[2,1],  -LC.lines_oc[2,::-1],  -LC.lines_L[2,-1]]
    mesh_model = create_loop(l_list, mesh_model, 15)

    # Wedge 
    
    index = find_line_index(LC.lines_S,CP.coord_sub,g_input.decoupling)
    index = index 
    buf_array = LC.lines_S[2,index:]
    buf_array = -buf_array 
    buf_array = buf_array[::-1]
    
    
    index = find_line_index(LC.lines_ch,CP.coord_channel,g_input.lt_d)
    index = index 
    buf_array2 = np.array(LC.lines_ch[2,index:])
    buf_array2 = -1*buf_array2 
    buf_array2 = buf_array2[::-1]

    
    l_list     = [-LC.lines_R[2,-1],-LC.lines_B[2,0],buf_array,LC.lines_base_ch[2,:],buf_array2,LC.lines_L_ov[2,:]]
    mesh_model = create_loop(l_list, mesh_model, 20)
    

    if g_input.cr !=0:
            
        index_a    = find_line_index(LC.lines_ch,CP.coord_channel,g_input.cr)
        index_b    = find_line_index(LC.lines_ch,CP.coord_channel,g_input.lt_d)-1
        buf_array = LC.lines_ch[2,index_a:index_b+1]
        buf_array = -buf_array[::-1]
        
        l_list     = [-LC.lines_R[2,2],-LC.lines_L_ov[2,:],buf_array,LC.lines_cr[2,:]]
        mesh_model = create_loop(l_list, mesh_model, 25)

        
        if g_input.lc !=0:
            a = []
            index_a    = find_line_index(LC.lines_ch,CP.coord_channel,(1-g_input.lc)*g_input.cr)
            index_b    = find_line_index(LC.lines_ch,CP.coord_channel,g_input.cr)-1

            buf_array = -LC.lines_ch[2,index_a:index_b+1]
            buf_array = buf_array[::-1]
            
            l_list = [-LC.lines_R[2,1],-LC.lines_cr[2,:],buf_array,LC.lines_lcr[2,:]]
            mesh_model = create_loop(l_list, mesh_model, 30)


            index_a    = 0
            index_b    = find_line_index(LC.lines_ch, CP.coord_channel,(1-g_input.lc)*g_input.cr)-1
            buf_array = -LC.lines_ch[2,index_a:index_b+1]
            buf_array = buf_array[::-1]

            l_list = [LC.lines_R[2,0],-LC.lines_lcr[2,:],buf_array,LC.lines_T[2,1]]
            mesh_model = create_loop(l_list, mesh_model, 35)

    index_a = find_line_index(LC.lines_ch,CP.coord_channel,g_input.lt_d)
    index = find_line_index(LC.lines_S,CP.coord_sub,g_input.lt_d)

    buf_array = -LC.lines_S[2,0:index]
    buf_array = buf_array[::-1]
    
    l_list = [LC.lines_ch[2,0:index_a], -LC.lines_ch_ov[2,:], buf_array, LC.lines_T[2,0]]
    mesh_model = create_loop(l_list, mesh_model, 40)
    
    
    index_a = find_line_index(LC.lines_S,CP.coord_sub,g_input.lt_d)
    index = find_line_index(LC.lines_S,CP.coord_sub,g_input.decoupling)
    buf_array = -LC.lines_S[2,index_a:index]
    buf_array = buf_array[::-1]
    index_b = find_line_index(LC.lines_ch,CP.coord_channel,g_input.lt_d)    
    
    l_list = [LC.lines_ch[2,index_b:],-LC.lines_base_ch[2,:],buf_array,LC.lines_ch_ov[2,:]]
    mesh_model = create_loop(l_list, mesh_model, 45)



    
    Left_side_of_subduction_surf   = gmsh.model.geo.addPlaneSurface([10],tag=100) # Left side of the subudction zone
    Oceanic_Crust_surf             = gmsh.model.geo.addPlaneSurface([15],tag=150) # Left side of the subudction zone
    Right_side_of_subduction_surf  = gmsh.model.geo.addPlaneSurface([20],tag=200) # Right side of the subudction zone    
    Lithhospheric_Mantle_surf      = gmsh.model.geo.addPlaneSurface([25],tag=250) # Right mantle
    Crust_LC_surf                  = gmsh.model.geo.addPlaneSurface([30],tag=300) # Crust LC
    Crust_UC_surf                  = gmsh.model.geo.addPlaneSurface([35],tag=350) # Crust LC
    Channel_surf_A                 = gmsh.model.geo.addPlaneSurface([40],tag=400) # Channel
    Channel_surf_B                 = gmsh.model.geo.addPlaneSurface([45],tag=450) # Channel

    
    mesh_model.geo.synchronize()

    mesh_model.addPhysicalGroup(2, [Left_side_of_subduction_surf],  tag=dict_surf['sub_plate'])
    mesh_model.addPhysicalGroup(2, [Oceanic_Crust_surf],            tag=dict_surf['oceanic_crust'])
    mesh_model.addPhysicalGroup(2, [Right_side_of_subduction_surf], tag=dict_surf['wedge'])
    mesh_model.addPhysicalGroup(2, [Lithhospheric_Mantle_surf],     tag=dict_surf['overriding_lm'])
    mesh_model.addPhysicalGroup(2, [Crust_LC_surf],                 tag=dict_surf['lower_crust'])
    mesh_model.addPhysicalGroup(2, [Crust_UC_surf],                 tag=dict_surf['upper_crust'])
    mesh_model.addPhysicalGroup(2, [Channel_surf_A],                tag=dict_surf['Channel_surf_a'])
    mesh_model.addPhysicalGroup(2, [Channel_surf_B],                tag=dict_surf['Channel_surf_b'])

    
    
    
    return mesh_model 

#----------------------------------------------------------------------------------------------------------------------
def create_gmesh(ioctrl):
    """_summary_: The function is composed by three part: -> create points, create lines, loop 
    ->-> 
    Args:
        ctrl (_type_): _description_
        ctrlio (_type_): _description_

    Returns:
        _type_: _description_
    """
    

    import gmsh 
    from   Subducting_plate import Slab
    import Function_make_mesh as fmm 


    g_input = geom_input()
    min_x           = g_input.x[0] # The beginning of the model is the trench of the slab
    max_x           = g_input.x[1]                 # Max domain x direction
    max_y           = g_input.y[1] 
    min_y           = g_input.y[0]                # Min domain y direction
    # Set up slab top surface a
    Data_Real = False; S = []
    van_keken = 0
    if (Data_Real==False) & (isinstance(S, Slab)== False):
        if van_keken == 1: 
            S = Slab(D0 = 100.0, L0 = 800.0, Lb = 400, trench = 0.0,theta_0=5, theta_max = 45.0, num_segment=100,flag_constant_theta=True,y_min=min_y)
        else: 
            S = Slab(D0 = 100.0, L0 = 800.0, Lb = 400, trench = 0.0,theta_0=1.0, theta_max = 45.0, num_segment=100,flag_constant_theta=False,y_min=min_y)

        for a in dir(S):
            if (not a.startswith('__')):
                att = getattr(S, a)
                if (callable(att)==False) & (np.isscalar(att)):
                    print('%s = %.2f'%(a, att))

    # Create the subduction interfaces using either the real data set, or the slab class
    slab_x, slab_y, bot_x, bot_y,oc_cx,oc_cy = fmm.function_create_slab_channel(Data_Real,g_input,SP=S)

    ind_oc_lc = np.where(slab_y == -g_input.cr*(1-g_input.lc))[0][0]
    ind_oc_cr = np.where(slab_y == -g_input.cr)[0][0]
    ind_oc_lt = np.where(slab_y == -g_input.lt_d)[0][0]
    
    
    fundamental_points = np.array([[slab_x[0],    slab_y[0]],                       # Point A [left corner global model] 
                                  [oc_cx[0],     oc_cy[1]],                        # Point B [oceanic crust]
                                  [bot_x[0],     bot_y[1]],                        # Point C [bottom lithosphere]
                                  [bot_x[-1],    bot_y[-1]],                       # Point D [bottom right node]
                                  [oc_cx[-1],    oc_cy[-1]],                       # Point E [bottom oceanic crust]
                                  [slab_x[-1],   slab_y[-1]],                      # Point F [slab top]
                                  [g_input.x[1], g_input.y[0]],                    # Point G
                                  [g_input.x[1], -g_input.lt_d],                   # Point H 
                                  [g_input.x[1], -g_input.cr],                     # Point I
                                  [g_input.x[1], -g_input.cr*(1-g_input.lc)],      # Point L 
                                  [g_input.x[1], -g_input.y[1]],                   # Point M 
                                  [oc_cx[ind_oc_lc],oc_cy[ind_oc_lc]],             # Point N
                                   [oc_cx[ind_oc_cr],oc_cy[ind_oc_cr]],            # Point O
                                  [oc_cx[ind_oc_lt],oc_cy[ind_oc_lt]]],dtype = np.float64) #           # Point P 
    
    mesh_model = create_gmsh(slab_x,slab_y,bot_x,bot_y,oc_cx,oc_cy,g_input,fundamental_points) 

    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.geo.mesh.setAlgorithm(2, dict_surf['Channel_surf_a'], 3)
    mesh_model.geo.mesh.setAlgorithm(2, dict_surf['Channel_surf_b'], 3)
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}


    mesh_model.mesh.generate(2)
    #mesh_model.mesh.setOrder(2)
    
    mesh_name = os.path.join(ioctrl.path_save,ioctrl.sname)
    
    gmsh.write("%s.msh"%mesh_name)
    
    return g_input
    
#------------------------------------------------------------------------------------------------------
def read_mesh(ioctrl,sc):



    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # 0, 1, ..., size-1
    size = comm.Get_size()  # total number of MPI processes


    mesh_name = os.path.join(ioctrl.path_save,'%s.msh'%ioctrl.sname)

    mesh, cell_markers, facet_markers = gmshio.read_from_msh(mesh_name, MPI.COMM_WORLD, gdim=2)
    
    
    if rank == 0: 
        # Read in mesh
        mesh_name = os.path.join(ioctrl.path_save,'%s.msh'%ioctrl.sname)
        msh = meshio.read(mesh_name)

        # Create and save one file for the mesh, and one file for the facets
        triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
        line_mesh = create_mesh(msh, "line", prune_z=True)
        meshio.write("mesh.xdmf", triangle_mesh)
        meshio.write("mt.xdmf", line_mesh)
        
        
    mesh = sc_f._scaling_mesh(mesh,sc)

    return mesh, cell_markers, facet_markers
#-----------------------------------------------------------------------------------------------------
def create_mesh_object(mesh,sc,ioctrl,g_input):    
    
    mesh, cell_markers, facet_markers = read_mesh(ioctrl, sc)
    
    Pph           = fem.functionspace(mesh, ("DG", 0))      # Material ID function space # Defined in the cell space {element wise} apperently there is a black magic that 
    # automatically does the interpolation of the function space, i.e. the function space is defined in the cell space, but it is automatically interpolated to the nodes                                    
    PT           = fem.functionspace(mesh, ("Lagrange", 2)) # Function space for the solution 
    # V-P 
    element_u = basix.ufl.element("Lagrange", "triangle", 2, shape=(2,))
    element_p = basix.ufl.element("DG", "triangle", 0) 
    V         = fem.functionspace(mesh,element_u)
    PD        = fem.functionspace(mesh,element_p)
    

    # Define the material property field
    phase        = fem.Function(Pph) # Create a function to hold the phase information
    phase.x.name = "phase"
    phase        = assign_phases(dict_surf, cell_markers, phase) # Assign phases using the cell tags and physical surfaces -> i.e. 10000 = Mantle ? is going to assign unique phase to each node? 
    # Correct the phase: 
    phase.x.array[:] -= 1 # Rather necessary remember to put plus one once you publish it 
    # -- 
    T_i             = fem.Function(PT)
    T_i.x.array[:]  = 0. 
    
    MESH = Mesh()
    
    
    MESH.mesh      = mesh 
    MESH.mesh_Ctag = cell_markers 
    MESH.mesh_Ftag = facet_markers 
    MESH.Pph       = Pph  
    MESH.PT        = PT 
    MESH.V         = V 
    MESH.PD        = PD 
    MESH.phase     = phase 
    MESH.T_i       = T_i 
    
    # dimension-> g_input
    g_input = g_input.dimensionless_ginput(sc)
    
    MESH.g_input   = g_input 
    


    return MESH



#------------------------------------------------------------------------------------------------------
def unit_test_mesh(ioctrl, sc):
    import numpy as np 
    import sys, os
    sys.path.append(os.path.abspath("src"))
    from numerical_control import IOControls

    
    
    g_input = geom_input() 
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # 0, 1, ..., size-1
    size = comm.Get_size()  # total number of MPI processes
    
    if rank == 0: 
        g_input = create_gmesh(ioctrl)
    
    M = create_mesh_object(mesh,sc,ioctrl, g_input)
    
    return M
    
#------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    ioctrl = IOControls(test_name = 'Debug_test',
                        path_save = '../Debug_mesh',
                        sname = 'Experimental')
    ioctrl.generate_io()
    
    sc = sc_f.Scal(L=660e3, Temp = 1350, eta = 1e21, stress = 1e9)
    
    unit_test_mesh(ioctrl,sc)
    
    

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # 0, 1, ..., size-1
    size = comm.Get_size()  # total number of MPI processes
    
    if rank == 0: 
        print('I am here') 
    

    

