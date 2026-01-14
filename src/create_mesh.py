# input for iFieldstone 

import os 
import sys
from .aux_create_mesh import Class_Points
from .aux_create_mesh import Class_Line
from .aux_create_mesh import create_loop
from .aux_create_mesh import find_line_index
from .numerical_control import IOControls, NumericalControls

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
from .scal import _scaling_mesh,Scal
import basix.ufl
from .utils import timing_function, print_ph
import dolfinx
from dolfinx.mesh import create_submesh
from .aux_create_mesh import Mesh, Domain, Class_Points, Class_Line, Geom_input, dict_tag_lines, dict_surf
from numpy.typing import NDArray


def debug_plot(target,global_line,global_point,color):
    for i in range(len(target)):
        line = np.abs(target[i])
        
        p0   = global_line[0,line-1]
        p1   = global_line[1,line-1]
        coord_x = [global_point[0,p0-1],global_point[0,p1-1]]            
        coord_y = [global_point[1,p0-1],global_point[1,p1-1]]
        plt.plot(coord_x,coord_y,c=color)
        


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

def assign_phases(dict_surf:dict, 
                  cell_tags:int,
                  phase:dolfinx.fem.Function)->dolfinx.fem.Function:
    """Assigns phase tags to the mesh based on the provided surface tags."""
    for tag, value in dict_surf.items():
        indices = cell_tags.find(value) 
        phase.x.array[indices] = np.full_like(indices,  value , dtype=PETSc.IntType)
    
    return phase 


def create_mesh_fenicsx(mesh, cell_type, prune_z=False):
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


def from_line_to_point_coordinate(L:int,LG:NDArray[np.int64],GP:NDArray[np.float64])->tuple[int,int,float,float]:
    p0   = LG[0,L-1]
    p1   = LG[1,L-1]

    coord_x = [GP[0,p0-1],GP[0,p1-1]]            
    coord_y = [GP[1,p0-1],GP[1,p1-1]]

    return p0, p1, coord_x, coord_y 




def create_domain_A(mesh_model, CP, LC, g_input):
    """
    Domain: subduction plate: domain composed by two area: oceanic crust and lithospheric mantle 
    Short disclaimer: I might be a bit retarded, but the way in which gmsh is assessing wheter or not 
    a line is correct, are still obscure. At the end I did trial and error till it works. My internal convection 
    is to start from the uppermost and rightmost corner and doing anticlockwise collection of lines 
    -> [the sign of the line depends on the order of the points, but I could not be arsed enough to create a function
    to recognise it]
    """
    if g_input.ocr != 0.0:
        
        l_list     = [LC.lines_oc[2,:],-LC.lines_B[2,-1],-LC.lines_BS[2,::-1], LC.lines_L[2,0]]
        mesh_model = create_loop(l_list, mesh_model, 10)

        # Oceanic crust  
        l_list     = [LC.lines_S[2,:], LC.lines_B[2,1], -LC.lines_oc[2,::-1],  -LC.lines_L[2,-1]]
        mesh_model = create_loop(l_list, mesh_model, 15)
    else:
        l_list     = [LC.lines_S[2,:],LC.lines_B[2,-1],-LC.lines_BS[2,::-1], -LC.lines_L[2,0]]  
        mesh_model = create_loop(l_list, mesh_model, 15)
        
        
    print('Finished to generate the curved loop for domain A [Subducting plate]')

    return mesh_model


def create_domain_B(mesh_model, CP, LC, g_input):
    
    index = find_line_index(LC.lines_S,CP.coord_sub,g_input.lt_d)
    index = index 
    buf_array = LC.lines_S[2,index:]
    buf_array = -buf_array 
    buf_array = buf_array[::-1]


    
    l_list     = [-LC.lines_R[2,-1],-LC.lines_B[2,0],buf_array, LC.lines_L_ov[2,:]]
    mesh_model = create_loop(l_list, mesh_model, 20)
    

    
    print('Finished to generate the curved loop for domain B [Wedge]')
    return mesh_model 


def create_domain_C(mesh_model, CP, LC, g_input):


    if g_input.cr !=0:
            


        
        if g_input.lc !=0:
            
            index_a    = find_line_index(LC.lines_S,CP.coord_sub,g_input.cr)
            index_b    = find_line_index(LC.lines_S,CP.coord_sub,g_input.lt_d)-1
            buf_array = LC.lines_S[2,index_a:index_b+1]
            buf_array = -buf_array[::-1]
        
            l_list     = [-LC.lines_R[2,2],-LC.lines_L_ov[2,:],buf_array,LC.lines_cr[2,:]]
            mesh_model = create_loop(l_list, mesh_model, 25)
            
            index_a    = find_line_index(LC.lines_S,CP.coord_sub,(1-g_input.lc)*g_input.cr)
            index_b    = find_line_index(LC.lines_S,CP.coord_sub,g_input.cr)-1

            buf_array = -LC.lines_S[2,index_a:index_b+1]
            buf_array = buf_array[::-1]
            
            l_list = [-LC.lines_R[2,1],-LC.lines_cr[2,:],buf_array,LC.lines_lcr[2,:]]
            mesh_model = create_loop(l_list, mesh_model, 35)


            index_a    = 0
            index_b    = find_line_index(LC.lines_S, CP.coord_sub,(1-g_input.lc)*g_input.cr)-1
            buf_array = -LC.lines_S[2,index_a:index_b+1]
            buf_array = buf_array[::-1]

            l_list = [LC.lines_R[2,0],-LC.lines_lcr[2,:],buf_array,LC.lines_T[2,:]]
            mesh_model = create_loop(l_list, mesh_model, 30)
        else:
            index_a    = find_line_index(LC.lines_S,CP.coord_sub,g_input.cr)
            index_b    = find_line_index(LC.lines_S,CP.coord_sub,g_input.lt_d)-1
            buf_array = LC.lines_S[2,index_a:index_b+1]
            buf_array = -buf_array[::-1]
        
            l_list     = [-LC.lines_R[2,1],-LC.lines_L_ov[2,:],buf_array,LC.lines_cr[2,:]]
            mesh_model = create_loop(l_list, mesh_model, 25)
            
            index_a    = 0
            index_b    = find_line_index(LC.lines_S, CP.coord_sub,g_input.cr)-1
            buf_array = -LC.lines_S[2,index_a:index_b+1]
            buf_array = buf_array[::-1]

            l_list = [LC.lines_R[2,0],-LC.lines_cr[2,:],buf_array,LC.lines_T[2,:]]
            mesh_model = create_loop(l_list, mesh_model, 30)
    else:   
        index    = find_line_index(LC.lines_S,CP.coord_sub,g_input.lt_d)-1
        buf_array = LC.lines_S[2,0:index+1]
        buf_array = -buf_array[::-1]
        
        l_list     = [-LC.lines_R[2,0],-LC.lines_L_ov[2,:],buf_array,LC.lines_T[2,:]]
        mesh_model = create_loop(l_list, mesh_model, 25)

    print('Finished to generate the curved loop for domain C [Crust]')
    return mesh_model


def create_physical_line(CP,LC, g_input,mesh_model):


    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}
    
    mesh_model.addPhysicalGroup(1, LC.tag_L_T, tag=dict_tag_lines['Top'])
    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    # Find point above the lithosphere 
    for i in range(len(LC.tag_L_R)):
        L = LC.tag_L_R[i]
        p0,p1,cx,cy = from_line_to_point_coordinate(L,LC.line_global, CP.global_points)
        if cy[0] == -g_input.lt_d or cy[1] == -g_input.lt_d:
            break 
    mesh_model.addPhysicalGroup(1, LC.tag_L_R[0:i+1], tag=dict_tag_lines['Right_lit'])
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}
    mesh_model.addPhysicalGroup(1, LC.tag_L_R[i+1:], tag=dict_tag_lines['Right_wed'])


    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}
    mesh_model.addPhysicalGroup(1, [LC.tag_L_B[0]], tag=dict_tag_lines['Bottom_wed'])

    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}
    mesh_model.addPhysicalGroup(1, LC.tag_L_B[1:], tag=dict_tag_lines['Bottom_sla'])
    
    
    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}
    mesh_model.addPhysicalGroup(1, LC.tag_L_Bsub, tag=dict_tag_lines['Subduction_bot'])

    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}
    mesh_model.addPhysicalGroup(1, LC.tag_L_L, tag=dict_tag_lines['Left_inlet'])

    for i in range(len(LC.tag_L_sub)):
        L = LC.tag_L_sub[i]
        p0,p1,cx,cy = from_line_to_point_coordinate(L,LC.line_global, CP.global_points)
        if cy[0] == -g_input.lt_d or cy[1] == -g_input.lt_d:
            break 
    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}
    mesh_model.addPhysicalGroup(1, LC.tag_L_sub[0:i+1],   tag=dict_tag_lines['Subduction_top_lit'])

    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}
    mesh_model.addPhysicalGroup(1, LC.tag_L_sub[i:],   tag=dict_tag_lines['Subduction_top_wed'])

    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}
    if g_input.ocr != 0.0:
        mesh_model.addPhysicalGroup(1, LC.tag_L_oc,    tag=dict_tag_lines['Oceanic'])
  
        mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}


    mesh_model.addPhysicalGroup(1, LC.tag_L_ov,    tag=dict_tag_lines['Overriding_mantle'])



    if g_input.cr !=0: 

        mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

        mesh_model.addPhysicalGroup(1,    LC.tag_L_cr, tag=dict_tag_lines['Crust_overplate'])

        if g_input.lc !=0:
            mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}
            mesh_model.addPhysicalGroup(1, LC.tag_L_Lcr,tag=dict_tag_lines['LCrust_overplate'])
    return mesh_model 

def create_gmsh(sx,        # subduction x
                sy,        # subdcution y 
                bsx,       # bottom subduction x
                bsy,       # bottom subduction y 
                oc_cx,     # oceanic cx 
                oc_cy,     # oceanic cu
                g_input):  # geometry input class 
    
    # -> USE GMSH FUNCTION 
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    mesh_model = gmsh.model()

    CP = Class_Points()
    mesh_model = CP.update_points(mesh_model,sx,sy,bsx,bsy,oc_cx,oc_cy,g_input)

    LC = Class_Line()
    mesh_model = LC.update_lines(mesh_model, CP, g_input)

    mesh_model = create_physical_line(CP,LC,g_input,mesh_model)

    mesh_model = create_domain_A(mesh_model, CP, LC, g_input)
    mesh_model = create_domain_B(mesh_model, CP, LC, g_input)
    mesh_model = create_domain_C(mesh_model, CP, LC, g_input)    
    

    # Wedge 
    
    
    if g_input.ocr != 0.0:
        Left_side_of_subduction_surf   = gmsh.model.geo.addPlaneSurface([10],tag=100) # Left side of the subudction zone
        Oceanic_Crust_surf             = gmsh.model.geo.addPlaneSurface([15],tag=150) # Left side of the subudction zone
    else:
        Left_side_of_subduction_surf   = gmsh.model.geo.addPlaneSurface([15],tag=150) # Left side of the subudction zone
    Wedge                          = gmsh.model.geo.addPlaneSurface([20],tag=200) # Right side of the subudction zone    
    Lithhospheric_Mantle_surf      = gmsh.model.geo.addPlaneSurface([25],tag=250) # Right mantle
    if g_input.cr !=0:
        Crust_UC_surf                  = gmsh.model.geo.addPlaneSurface([30],tag=300) # Crust LC
        if g_input.lc !=0:
            Crust_LC_surf                  = gmsh.model.geo.addPlaneSurface([35],tag=350) # Crust LC
    

    
    mesh_model.geo.synchronize()

    mesh_model.addPhysicalGroup(2, [Left_side_of_subduction_surf],  tag=dict_surf['sub_plate'])
    if g_input.ocr != 0.0:
        mesh_model.addPhysicalGroup(2, [Oceanic_Crust_surf],            tag=dict_surf['oceanic_crust'])
    mesh_model.addPhysicalGroup(2, [Wedge],                         tag=dict_surf['wedge'])
    mesh_model.addPhysicalGroup(2, [Lithhospheric_Mantle_surf],     tag=dict_surf['overriding_lm'])
    if g_input.cr !=0:
        mesh_model.addPhysicalGroup(2, [Crust_UC_surf],                 tag=dict_surf['upper_crust'])
        if g_input.lc !=0:
            mesh_model.addPhysicalGroup(2, [Crust_LC_surf],                 tag=dict_surf['lower_crust'])


    return mesh_model 

#----------------------------------------------------------------------------------------------------------------------
def create_gmesh(ioctrl   :IOControls,
                 g_input  :Geom_input,
                 ctrl     :NumericalControls):
    """_summary_: The function is composed by three part: -> create points, create lines, loop 
    ->-> 
    Args:
        ctrl (_type_): _description_
        ctrlio (_type_): _description_

    Returns:
        _type_: _description_
    """
    

    import gmsh 
    from   .Subducting_plate import Slab
    from  .aux_create_mesh import function_create_slab_channel 



    min_x           = g_input.x[0]        # The beginning of the model is the trench of the slab
    max_x           = g_input.x[1]                 # Max domain x direction
    max_y           = g_input.y[1] 
    min_y           = g_input.y[0]                # Min domain y direction
    # Set up slab top surface a
    Data_Real = False; S = []
    van_keken =ctrl.van_keken
    if (Data_Real==False) & (isinstance(S, Slab)== False):
        if van_keken == 1: 
            min_x           =0.0 # The beginning of the model is the trench of the slab
            max_x           = 660e3                 # Max domain x direction
            max_y           = 0.0
            min_y           = -600.0e3               # Min domain y direction            
            g_input.x[0]   = min_x
            g_input.x[1]   = max_x
            g_input.y[1]   = max_y
            g_input.y[0]   = min_y
        
            S = Slab(D0 = 100.0, L0 = 800.0, Lb = 800, trench = 0.0,theta_0=5, theta_max = 45.0, num_segment=100,flag_constant_theta=True,y_min=min_y)
            
        else: 
            S = Slab(D0 = 100.0, L0 = 800.0, Lb = 800, trench = 0.0,theta_0=1.0, theta_max = 45.0, num_segment=100,flag_constant_theta=False,y_min=min_y)


        for a in dir(S):
            if (not a.startswith('__')):
                att = getattr(S, a)
                if (callable(att)==False) & (np.isscalar(att)):
                    print('%s = %.2f'%(a, att))

       # Max domain x direction

    # Create the subduction interfaces using either the real data set, or the slab class
    slab_x, slab_y, bot_x, bot_y,oc_cx,oc_cy = function_create_slab_channel(Data_Real,g_input,SP=S)
    if slab_x[-1]>g_input.x[1]:
        print_ph('Shortcoming: the slab is out of the domain, please increase the domain size, I add 60 km to the domain along x direction, fear not')
        g_input.x[1] = slab_x[-1]+60e3
        max_x        = g_input.x[1]
        
    min_x           = g_input.x[0] # The beginning of the model is the trench of the slab
    max_x           = g_input.x[1]          
    if g_input.cr != 0:
        ind_oc_cr = np.where(slab_y == -g_input.cr)[0][0]
        if g_input.lc !=0:     
            ind_oc_cr = np.where(slab_y == -g_input.cr)[0][0]
        else: 
            ind_oc_lc = []
    else: 
        ind_oc_cr = []
        ind_oc_lc = []
        
    ind_oc_lt = np.where(slab_y == -g_input.lt_d)[0][0]
    
    
    mesh_model = create_gmsh(slab_x,slab_y,bot_x,bot_y,oc_cx,oc_cy,g_input) 

    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}
    theta = np.arctan2((slab_y[-1]-slab_y[-2]),(slab_x[-1]-slab_x[-2]))
    g_input.theta_out_slab = theta   # Convert to degrees

    mesh_model.mesh.generate(2)
    #mesh_model.mesh.setOrder(2)
    
    mesh_name = os.path.join(ioctrl.path_save,ioctrl.sname)
    
    gmsh.write("%s.msh"%mesh_name)
    
    
    return g_input


def extract_facet_boundary(Mesh, Mfacet_tag, submesh, sm_vertex_maps, boundary,m_id):
    """
    ]Function to create an array of list containing the facet of BC (from the parent mesh)

    """

    # Extract facet from the parent mesh. Boundary -> list of marker of the boundary [i.e., {5,6}].     
    chosen_one = []
    for tag in boundary:
        chosen_one.extend(Mfacet_tag.find(tag))
        
    #--     
     # Parent Mesh -> extract information. 
     # First extract information for any particular boundary (also composite), then select the facet that are belonging to this particular boundary 
     # then extract the nodes id {the function that creates the submesh release the maps to the parent mesh, we can exploit it for generating a list of
     # facet}
    #--     
    
    # Remove the double entries []    
    chosen_one = np.unique(chosen_one)
    # Create the topology => from the set of facet, generate an object containing number of facet and the point/vertices that define each of them
    Mesh.topology.create_connectivity(Mfacet_tag.dim, 0)
    # Extract connectivity
    facet_to_vertex = Mesh.topology.connectivity(Mfacet_tag.dim, 0)
    # From facet number to node 
    vertex_indices = [] # Empty list
    for f in chosen_one:
        vertex_indices.extend(facet_to_vertex.links(f))

    # Remove double entries
    vertex_indices = np.unique(vertex_indices)
    # -- 
    # Sub mesh 
    # --
    # Extract the vertex from submesh 
    cell_dim = submesh.topology.dim
    cell_to_vertex = submesh.topology.connectivity(cell_dim, 0)

    # Extract the vertices
    v_ids = cell_to_vertex.links(0)

    # Extract the indices and then find the unique one 
    all_vertex_ids = []
    for c in range(submesh.topology.index_map(cell_dim).size_local):
        all_vertex_ids.extend(cell_to_vertex.links(c))

    unique_vertex_ids = np.unique(all_vertex_ids)

    # Loop over the indeces of the nodes belonging to the given boundary from the parent mesh
    # -> find the index in the local sub-mesh mapping 
    # -> Collect the index from the local_mesh index 
    ind_facet = []
    for i in range(len(vertex_indices)): 
        ind = np.where(sm_vertex_maps == vertex_indices[i])[0]
        ind_facet.extend(unique_vertex_ids[ind])
    

    # 3. Loop over all the facet -> extract the node ids of the facet of the submesh if both of them exist in the array ind_facet
    # -> collect the number of the facet and collect into a list -> then turn into an array -> you have your boundary 
    connectivity = submesh.topology.connectivity(1, 0)   
    chosen_facet = []        
    for facet_index in range(submesh.topology.index_map(1).size_local):
        sub_mesh_vertex_index = connectivity.links(facet_index)
        if (np.isin(sub_mesh_vertex_index[0],ind_facet)) and (np.isin(sub_mesh_vertex_index[1],ind_facet)): 
            chosen_facet.append(facet_index)
    
    chosen_facet = np.asarray(chosen_facet,dtype=np.int32)
    values       = np.full(chosen_facet.size,m_id,dtype = np.int32)
    

    return  chosen_facet, values


@timing_function
def create_subdomain(mesh, mesh_tag, facet_tag, phase_set, name, phase):
    from dolfinx.mesh import meshtags

    def facet_BC(mesh, facet_tag, submesh, vertex_maps, specs):
        chosen_total = []
        val_total = []
        for boundary, m_id in specs:
            ch_f, val = extract_facet_boundary(mesh, facet_tag, submesh, vertex_maps, boundary, m_id)
            chosen_total.extend(ch_f)
            val_total.extend(val)
        fac = np.asarray(chosen_total, dtype=np.int32)
        val = np.asarray(val_total, dtype=np.int32)
        FT = meshtags(submesh, 1, fac, val)
        return FT



    """
    https://fenicsproject.discourse.group/t/how-to-define-bcs-on-boundaries-of-submeshes-of-a-parent-mesh/5470/3
    """
    # Find the chosen markers 
    marked_cells = []
    
    for marker in phase_set:
        marked_cells.extend(mesh_tag.find(marker))

    marked_cells = np.array(marked_cells, dtype=np.int32)
    # 
    submesh, entity_maps, vertex_maps, node_maps = create_submesh(mesh, mesh.topology.dim, marked_cells)
    
    submesh.topology.create_connectivity(2, 0)
    submesh.topology.create_connectivity(1, 0)
    submesh.topology.create_connectivity(2, 1)
    submesh.topology.create_connectivity(1, 2)
    
    # Facet marker original mesh -> 
    # Select subduction: 
    if name == 'Subduction': 
        # top subduction 8-9 For the subduction subdomain, the tag of the subdcution 
        # are the entire top surface 
        # -- 
        specs = [([8, 9],1), ([6],2), ([7],3), ([5],4)] # [8,9] are the top subduction[6] is the bottom subduction, [7] is the left side of the subduction, [5] is the right side of the subduction
        dict_local = {
            'top_subduction' : 1, # Top subduction
            'bot_subduction' : 2, # Right side of the subduction
            'inflow'         : 3, # Bottom side of the subduction
            'outflow'        : 4, # Left side of the subduction
        }
        # --
        bc = facet_BC(mesh, facet_tag, submesh, vertex_maps, specs)
        # -- 
    elif name == 'Wedge':
        
        specs = [([11],1), ([4],2), ([3],3), ([9],4)] # [1] is the top, [2] is the right side of the wedge, [3] is the bottom side of the wedge, [4] is the left side of the wedge
        dict_local = {
            'overriding'    : 1, # Top subduction
            'right'         : 2, # Right side of the subduction
            'bottom'        : 3, # Bottom side of the subduction
            'slab'          : 4, # Left side of the subduction
        }
        bc = facet_BC(mesh, facet_tag, submesh, vertex_maps, specs)

    elif name == 'Lithosphere':
        
        specs = [([11],1), ([1],2), ([2],3), ([8],4)] # [1] is the top, [2] is the right side of the wedge, [3] is the bottom side of the wedge, [4] is the left side of the wedge
        dict_local = {
            'overriding'    : 1, # Top subduction
            'right'         : 2, # Right side of the subduction
            'bottom'        : 3, # Bottom side of the subduction
            'slab'          : 4, # Left side of the subduction
        }

        bc = facet_BC(mesh, facet_tag, submesh, vertex_maps, specs)

    # Create the functionsubspace for the subdomain
    
    Sol_Spaceph         = fem.functionspace(submesh, ("DG", 0))      # Material ID function space # Defined in the cell space {element wise} apperently there is a black magic that

    
    ph = fem.Function(Sol_Spaceph)
    ph.x.name = "phase"
    
    
    ph.interpolate(phase, cells0=entity_maps, cells1=np.arange(len(entity_maps)))
    
    domain = Domain(hierarchy = 'Child', mesh = submesh, cell_par = entity_maps, node_par = vertex_maps, facets = bc , phase = ph,solPh = Sol_Spaceph , bc_dict = dict_local)
    
    return domain


    
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

        
        pt_save = ioctrl.path_save
        if not os.path.exists(pt_save):
            os.makedirs(pt_save)
    
        pt_save = os.path.join(pt_save, ioctrl.sname)
        if not os.path.exists(pt_save):
            os.makedirs(pt_save)    
        


        # Create and save one file for the mesh, and one file for the facets
        triangle_mesh = create_mesh_fenicsx(msh, "triangle", prune_z=True)
        line_mesh = create_mesh_fenicsx(msh, "line", prune_z=True)
        meshio.write("%s/mesh.xdmf"%pt_save, triangle_mesh)
        meshio.write("%s/mt.xdmf"%pt_save, line_mesh)
        
        
    mesh = _scaling_mesh(mesh,sc)

    return mesh, cell_markers, facet_markers
#-----------------------------------------------------------------------------------------------------
def create_shear_heating_interface(mesh,cell_markers, facet_markers):
    
    new = facet_markers.find(9)  # Find the facet with tag 9 -> subduction top wedge
    
    return new_facet

#------------------------------------------------------------------------------------------------------
@timing_function
def create_mesh_object(mesh,sc,ioctrl,g_input):    
    
    from dolfinx.mesh import create_submesh

    mesh, cell_markers, facet_markers = read_mesh(ioctrl, sc)
    
    Pph           = fem.functionspace(mesh, ("DG", 0))      # Material ID function space # Defined in the cell space {element wise} apperently there is a black magic that 

    # Define the material property field

    phase           = fem.Function(Pph) # Create a function to hold the phase information
    phase.x.name    = "phase"
    phase           = assign_phases(dict_surf, cell_markers, phase) # Assign phases using the cell tags and physical surfaces -> i.e. 10000 = Mantle ? is going to assign unique phase to each node? 

    # Correct the phase: 
    phase.x.array[:] -= 1 # Rather necessary remember to put plus one once you publish it 

    #-- Create additional facet for the shear heating. Since this hell is requiring a lot of useless work, 
    #-- I need to create a yet another ad hoc function for this. 
    
    


    # -- 
    domainG = Domain(
            mesh=mesh,
            facets=facet_markers,
            Tagcells=cell_markers,
            phase=phase,
            solPh=Pph,
            bc_dict=dict_tag_lines,
            )    
    
    domainA = create_subdomain(mesh, cell_markers, facet_markers, [1,2]  , 'Subduction',  phase)
    
    domainB = create_subdomain(mesh, cell_markers, facet_markers, [3]    , 'Wedge',       phase)
    
    domainC = create_subdomain(mesh, cell_markers, facet_markers, [4,5,6], 'Lithosphere', phase)

    # --
    
    MESH = Mesh()
    
    MESH.domainG              = domainG
    MESH.domainA              = domainA     
    MESH.domainB              = domainB
    MESH.domainC              = domainC   

    # dimension-> g_input
    g_input                   = g_input.dimensionless_ginput(sc)
    
    MESH.g_input              = g_input 
    
    return MESH



#------------------------------------------------------------------------------------------------------
def unit_test_mesh(ioctrl, sc,g_input,ctrl):
    import numpy as np 
    import sys, os
    sys.path.append(os.path.abspath("src"))
    from numerical_control import IOControls
    
    print_ph("[] - - - -> Creating mesh <- - - - []")

        
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # 0, 1, ..., size-1
    size = comm.Get_size()  # total number of MPI processes
    
    if rank == 0: 
        g_input = create_gmesh(ioctrl,g_input,ctrl)
    
    M = create_mesh_object(mesh,sc,ioctrl, g_input)
    M.comm = comm 
    M.rank = rank 
    M.size = size 

    return M
    
#------------------------------------------------------------------------------------------------

def create_mesh(ioctrl, sc, g_input,ctrl):
    import numpy as np 
    import sys, os
    sys.path.append(os.path.abspath("src"))
    from .numerical_control import IOControls
    

        
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # 0, 1, ..., size-1
    size = comm.Get_size()  # total number of MPI processes
    
    if rank == 0: 
        g_input = create_gmesh(ioctrl,g_input,ctrl)
    
    M = create_mesh_object(mesh,sc,ioctrl, g_input)
    M.comm = comm 
    M.rank = rank 
    M.size = size 


    return M
    
#------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    
    ioctrl = IOControls(test_name = 'Debug_test',
                        path_save = '../Debug_mesh',
                        sname = 'Experimental')
    ioctrl.generate_io()
    
    sc = Scal(L=660e3, Temp = 1350, eta = 1e21, stress = 1e9)
    
    g_input = Geom_input(x = np.array([0.,660e3]),
                         y = np.array([-600e3,0.]),
                         cr=20e3,
                         ocr=6e3,
                         lit_mt=30e3,
                         lc = 0.5,
                         wc = 2.0e3,
                         slab_tk = 130e3, 
                         decoupling = 100e3)
    
    ctrl = NumericalControls()
    ctrl.van_keken = 1
    
    M = unit_test_mesh(ioctrl,sc, g_input,ctrl)
    
    
    

  
    
    

    

