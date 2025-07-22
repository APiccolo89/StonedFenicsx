# input for iFieldstone 
import make_mesh.Create_Mesh as C_msh
import os
from material_property.phase_db import PhaseDataBase
from solver_function.numerical_control import NumericalControls
from material_property.phase_db import _generate_phase
from solver_function.numerical_control import IOControls
from solver_function.numerical_control import ctrl_LHS
from numba.experimental import jitclass
from solver_function.numerical_control import dict_k, dict_Cp, dict_rho, dict_rheology
import numpy as np
from solver_function.numerical_control import bc_controls
from solver_function.scal import Scal 
import ufl
import numpy
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


dict_surf = {
    'sub_plate'         : 1,
    'oceanic_crust'     : 2,
    'wedge'             : 3,
    'overriding_lm'     : 4,
    'lower_crust'       : 5,
    'upper_crust'       : 6,
    'Channel_surf_a'    : 7,
    'Channel_surf_b'    : 8,
}


def assign_phases(dict_surf, cell_tags,phase):
    """Assigns phase tags to the mesh based on the provided surface tags."""
    for tag, value in dict_surf.items():
        indices = cell_tags.find(value) 
        phase.x.array[indices] = np.full_like(indices,  value , dtype=PETSc.IntType)
    
    return phase 

def get_line_points(gmsh,line_ids):
    gmsh.model.geo.synchronize()
    for lid in line_ids:
        pts = gmsh.model.getAdjacencies(1, abs(lid))[1]
        print(f"Line {lid}: {pts[0]} → {pts[1]}")
    return line_ids

# Prepare a few small function. By the end I need to release array because list in python are a scam. 

def find_tag_line(coord,x,dir):
    if dir == 'x':
        a = 0 
    else:
        a = 1 
    
    i = np.where(coord[a,:]==-x)
    i = i[0][0];i = coord[2,i]
    return np.int32(i)                            

class c_phase():
    def __init__(self,
                 cr=35e3,
                 ocr=6e3,
                 lit_mt=70e3,
                 lc = 0.5,
                 wc = 1e5):
        self.cr = cr 
        self.ocr = ocr
        self.lit_mt = lit_mt 
        self.lc = lc
        self.wc = wc 
        self.lt_d = -(cr+lit_mt)
        

def find_line_index(Lin_ar,point,d):
    
    for i in range(len(Lin_ar[0,:])-1):
        # Select pint of the given line
        p0 = np.int32(Lin_ar[0,i])

        p1 = np.int32(Lin_ar[1,i])
        # find the index of the point 
        ip0 = np.where(p0==np.int32(point[2,:]))
        ip = np.where(p1==np.int32(point[2,:]))
        print(i,p0,p1)
        # Check wheter or not the coordinate z is the one. 
        z1 = point[1,ip]
        if z1 == -d: 
            X = [point[0,ip0],point[0,ip]]
            Y = [point[1,ip0],point[1,ip]]
            print(X)
            print(Y)
            index = i+1 
            break    
    
    
    return index 


def set_initial_():
    v_ST = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]

    n_phase = 3 
    v_s = [5.0,0.0]
    v_s_arr = np.asarray(v_s, dtype=np.float64)
    slab_age = 50.e6   # slab age [yr]

    X = [0,1000e3]
    Z = [-600e3,0]

    # ========== Bending function ========== #

    option_k_ = dict_k['Tosi2013']
    option_Cp_ = dict_Cp['base']
    option_rho_ =dict_rho['pres_dep']
    rheology_ = dict_rheology['default'] # 0: default, 1: diffusion only, 2: composite, 3: dislocation only, 4: composite iter
    r_channel = 2000
    s_ = '_SUPG_Steady_State_relax'

    t_name = 'T_ok%d_Cp%d_r%d_v%d_vel%d_R_%d_%s'%(option_k_,option_Cp_,option_rho_,rheology_,int(v_s[0]),int(r_channel),s_)
    p_save = '../time_crank'

    sname        = t_name

    v_s=np.asarray(v_s, dtype=np.float64)
    scal = Scal(L=660e3,stress = 1e9,eta = 1e22,Temp=1350)
    # ========== constants ========== #
    num_ctrl = NumericalControls(option_k = option_k_,
                            option_Cp = option_Cp_,
                            option_rho = option_rho_, 
                            rheology = rheology_,
                            it_max = 100,
                            tol    = 1e-4,
                            relax  = 0.8,
                            eta_max = 1e26, 
                            eta_def = 1e21, 
                            eta_min = 1e18,
                            Ttop    = 0.0,
                            Tmax    = 1300.0,
                            g = 9.81, 
                            pressure_scaling=1e22/1000e3,
                            slab_age = slab_age,
                            v_s = v_s,
                            advect_temperature = 1,
                            pressure_bc = 0,
                            petsc = 1,
                            dt = 1.0,
                            time_dependent_v = 1)

    num_ctrl.Ttop = num_ctrl.Ttop/scal.Temp 
    num_ctrl.Tmax = num_ctrl.Tmax/scal.Temp
    num_ctrl.g    = num_ctrl.g/scal.ac
    num_ctrl.v_s    = num_ctrl.v_s/(scal.L/scal.T)
    num_ctrl.pressure_scaling = (num_ctrl.pressure_scaling*scal.L)/scal.eta
    num_ctrl.t_max = 20e6 * num_ctrl.scal_year/scal.T # 20 Myr in seconds



    IOCtrl = IOControls(test_name = t_name,
                        path_save = p_save,
                        sname = sname)
    IOCtrl.generate_io()

    lhs_temp = ctrl_LHS(dz=1e3,
                        nz=107,
                        end_time = 200e6,
                        dt = 5e3,
                        recalculate = 1,
                        van_keken = 1,)
                        #d_RHS = -100e3)
    lhs_temp._scale_parameters(num_ctrl)
    lhs_temp.d_RHS /= scal.L 

    make_mesh = True
    meshfilename = '%s.msh'%(sname)

    BC_STOKES = [['Top','NoSlip',np.array([0.0,0.0],dtype=np.float64)],
                 ['Bottom','DoNothing',np.array([0.0,0.0],dtype=np.float64)],
                 ['Right','DoNothing',np.array([0.0,1.0],dtype=np.float64)],
                 ['Left','DoNothing',np.array([0.0,1.0],dtype=np.float64)],
                 ['Feature_on']]
    BC_ENERGY = [['Top','Isothermal',np.array([num_ctrl.Ttop,0.0],dtype=np.float64)],
                 ['Bottom','Open',np.array([num_ctrl.Tmax,0.0],dtype=np.float64)],
                 ['Right','Open',np.array([0.0,0.0],dtype=np.float64)],
                 ['Left','Open',np.array([0.0,0.0],dtype=np.float64)],
                 ['Feature_on']]
    BC_spec = bc_controls(BC_STOKES,BC_ENERGY)

    # model setup

    PDB = PhaseDataBase(2)
    _generate_phase(PDB,2,id=0,name_diffusion='Van_Keken_diff',name_dislocation='Van_Keken_disl')
    _generate_phase(PDB,2,id=1,name_diffusion='Van_Keken_diff',name_dislocation='Van_Keken_disl')

    #_generate_phase(PDB,2,id=1,name_diffusion='Hirth_Wet_Olivine_diff',name_dislocation='Hirth_Wet_Olivine_disl')
    PDB.Vdis[0] = 0e-6 # cm3/mol
    PDB.Vdif[0] = 0e-6 # cm3/mol

    PDB.Vdis[1] = 0e-6 # cm3/mol
    PDB.Vdif[1] = 0e-6 # cm3/mol
    
    return PDB,BC_spec,num_ctrl,scal,IOCtrl


def _create_points(mesh,                                            # I am not able to classify 
                   x:float,                                         # coordinate point/points
                   y:float,                                         # coordinate point/points
                   res:float,                                       # resolution of the ppint
                   tag_pr:int,                                      # maximum tag of the previous group
                   point_flag:bool=False)-> tuple[int, int, float]: # a flag to highlight that there is only one point, yes, lame. 
    
    """_summary_: Summary: function that creates the point using gmsh. 
    gmsh   : the gmsh object(?)
    x,y    : coordinate [float]-> can be an array as well as a single point
    res    : the resolution of the triangular mesh 
    tag_pr : previous tag -> i.e., since gmsh is internally assign a tag, I need to keep track of these points for then updating coherently
    Returns:
        max_tag  : -> the maximum tag of function call
        tag_list : -> the list of tag for the point of this function call
        coord    : -> the coordinates -> rather necessary for some functionality later on. Small modification {should I change the name?} -> Add also the tag,  
    
    Problem that solves: for a given list of points (from 1 to N) -> call gmsh function, create points, store information for the setup generation
    -> Nothing sideral, but I try to document everything for the future generations. 
    """
    
    
    tag_list = []

    if point_flag == True:
        coord = np.zeros([3,1],dtype=np.float64)
        mesh.geo.addPoint(x, y, 0.0,res,tag_pr+1) 
        tag_list.append(tag_pr+1)
        coord[0] = x 
        coord[1] = y
        coord[2] = tag_pr+1
        
    else:
        coord = np.zeros([3,len(x)],dtype=np.float64)
        for i in range(len(x)):

            mesh.geo.addPoint(x[i], y[i], 0.0, res, tag_pr + 1 + i) 
            tag_list.append(tag_pr+1+i)
            coord[0,i]   = x[i]
            coord[1,i]   = y[i] 
            coord[2,i]   = tag_pr + 1 + i
            
    
    max_tag = np.max(tag_list)

    
    return max_tag,tag_list,coord,mesh


def _create_lines(mesh_model,previous, tag_p,flag=False):
    
    len_p = len(tag_p)-1
    tag_l = []
    previous = previous+1
    lines = np.zeros([3,len_p],dtype=np.int32)
    for i in range(len_p): 
        a = tag_p[i]
        b = tag_p[i+1]
        'it appears that the gmsh is an horrible person: i try to give the golden rule that the tag of the points are following an order,'
        'which entails that the minimum of the two point defining a segment is the first -> such that it is always possible determine the fucking order'
 
        tag_1 = np.min([a,b])
        tag_2 = np.max([a,b]) 
        mesh_model.geo.addLine(tag_1,tag_2,previous+i)
        tag_l.append(previous+i)
        lines[0,i] = tag_1 
        lines[1,i] = tag_2 
        lines[2,i] = previous+i
        
    max_tag = np.max(tag_l)
    
    return max_tag,tag_l,lines,mesh_model


# First draft -> I Need to make it a decent function, otherwise this is a fucking nightmare
def create_parallel_mesh(ctrl,ctrlio):
    """_summary_: The function is composed by three part: -> create points, create lines, loop 
    ->-> 
    Args:
        ctrl (_type_): _description_
        ctrlio (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    
    from dolfinx.io import XDMFFile, gmshio
    from mpi4py import MPI
    import gmsh 
    from make_mesh.Subducting_plate import Slab
    import make_mesh.Function_make_mesh as fmm 


    c_phase = fmm.c_phase()
    min_x           = 0.0 # The beginning of the model is the trench of the slab
    max_x           = 1000e3                 # Max domain x direction
    max_y           = 0.0 
    min_y           = -660e3                # Min domain y direction
    # Set up slab top surface a
    Data_Real = False; S = []
    van_keken = 0
    if (Data_Real==False) & (isinstance(S, Slab)== False):
        if van_keken == 1: 
# Max domain y direction 
            S = Slab(D0 = 100.0, L0 = 800.0, Lb = 400, trench = 0.0,theta_0=5, theta_max = 45.0, num_segment=100,flag_constant_theta=True,y_min=min_y)
        else: 
            S = Slab(D0 = 100.0, L0 = 800.0, Lb = 400, trench = 0.0,theta_0=1.0, theta_max = 45.0, num_segment=100,flag_constant_theta=False,y_min=min_y)

        for a in dir(S):
            if (not a.startswith('__')):
                att = getattr(S, a)
                if (callable(att)==False) & (np.isscalar(att)):
                    print('%s = %.2f'%(a, att))

    # Create the subduction interfaces using either the real data set, or the slab class
    slab_x, slab_y, theta_mean,channel_x,channel_y,extra_x,extra_y,isch,oc_cx,oc_cy = fmm.function_create_slab_channel(Data_Real,c_phase,SP=S)


    # -> USE GMSH FUNCTION 
    gmsh.initialize()

    gmsh.model.geo.synchronize()

    mesh_model = gmsh.model()
    # Function CREATE POINTS
    
    #-- Create the subduction,channel and oceanic crust points 
    # Point of subduction -> In general, the good wisdom would tell you to not put a disordered list of points, 
    # but, I am paranoid, therefore: just create the geometry of slab, channel and oceanic crust such that y = f(x) where f(x) is always growing (or decreasing)
    # if you want to be creative, you have to modify the function of create points, up to you, no one is against it. 
    
    max_tag_s,tag_subduction,coord_sub,mesh_model            = _create_points(mesh_model,slab_x,slab_y,c_phase.wc,0)
    max_tag_c,tag_channel,coord_channel,mesh_model            = _create_points(mesh_model,channel_x,channel_y,c_phase.wc,max_tag_s)
    max_tag_oc,tag_oc,coord_ocean,mesh_model                  = _create_points(mesh_model,oc_cx,oc_cy,c_phase.wc,max_tag_c)
    # -- Here are the points at the boundary of the model. The size of the model is defined earlier, and subduction zone is modified as such to comply the main geometrical input, 
    # I used subduction points because they define a few important point. 
    max_tag_a,tag_left_c,coord_lc,mesh_model                  = _create_points(mesh_model,min_x,min_y,c_phase.wc*2,max_tag_oc,True)
    max_tag_b,tag_right_c_b,coord_bc,mesh_model               = _create_points(mesh_model,max_x,min_y,c_phase.wc*2,max_tag_a,True)
    max_tag_c,tag_right_c_l,coord_lr,mesh_model               = _create_points(mesh_model,max_x,-c_phase.lt_d,c_phase.wc*2,max_tag_b,True)
    max_tag_d,tag_right_c_t,coord_top,mesh_model              = _create_points(mesh_model,max_x,max_y,c_phase.wc*2,max_tag_c,True)
    
    if c_phase.cr !=0: 
        max_tag_e,tag_right_c_cr,coord_crust,mesh_model       = _create_points(mesh_model,max_x,-c_phase.cr,c_phase.wc*2,max_tag_d,True)
        if c_phase.lc !=0: 
            max_tag_f,tag_right_c_lcr,coord_lcr,mesh_model    = _create_points(mesh_model,max_x,-c_phase.cr*(1-c_phase.lc),c_phase.wc*2,max_tag_e,True)
    
    global_points = np.hstack([coord_sub,coord_channel,coord_ocean,coord_lc,coord_bc,coord_lr,coord_top,coord_crust,coord_lcr])
    
    
    # Function CREATE LINES
    
    #-- Lines  
    # This part of the script is devoted to create the line defining the main unit {Line as entity not physical line}

    # -- Create Boundary Lines [top boundary] 
    p_list = [tag_subduction[0],tag_channel[0],tag_right_c_t[0]]
    max_tag_top,tag_L_T,lines_T,mesh_model = _create_lines(mesh_model,0,p_list,False)
    
    
    
    #[right boundary]
    # -- This a tedius job, but remember, dear, to just follow the logic: since I introduced a few new functionality, and the gmsh function are pretty annoying 
    # I tried to make the function as simple as my poor demented mind can do. So, it is not the most easiest possible, but the easiest that I was able to conceive
    # in 10 minutes. Good luck 
    
    if c_phase.cr !=0: 
        if c_phase.lc !=0:
            p_list = [tag_right_c_t[0],tag_right_c_lcr[0],tag_right_c_cr[0],tag_right_c_l[0],tag_right_c_b[0]]
        else:
            p_list = [tag_right_c_t[0],tag_right_c_cr[0],tag_right_c_l[0],tag_right_c_b[0]]
    else: 
            p_list = [tag_right_c_t[0],tag_right_c_l[0],tag_right_c_b[0]]
    
    max_tag_right,tag_L_R,lines_R,mesh_model = _create_lines(mesh_model,max_tag_top,p_list,False)
    #[bottom boundary]
    p_list = [tag_right_c_b[0],tag_subduction[-1],tag_oc[-1],tag_left_c[0]]
    max_tag_bottom,tag_L_B,lines_B,mesh_model = _create_lines(mesh_model,max_tag_right,p_list,False)
    # -- 
    p_list = [tag_left_c[0],tag_oc[0],tag_subduction[0]]
    max_tag_left,tag_L_L,lines_L,mesh_model = _create_lines(mesh_model,max_tag_bottom,p_list,False)
    # -- Create Lines 
    max_tag_line_subduction,tag_L_sub,lines_S,mesh_model  = _create_lines(mesh_model,max_tag_left,tag_subduction,False)
    max_tag_line_channel,tag_L_ch,lines_ch,mesh_model     = _create_lines(mesh_model,max_tag_line_subduction,tag_channel,False)
    max_tag_line_ocean,tag_L_oc,lines_oc,mesh_model       = _create_lines(mesh_model,max_tag_line_channel,tag_oc,False)
    # Create line overriding plate:
    #-- find tag of the of the channel # -> find the mistake: confusion with index types
    i_s = find_tag_line(coord_sub,c_phase.lt_d,'y')

    i_c = find_tag_line(coord_channel,c_phase.lt_d,'y')
    # CHECK!
    p_list = [i_s,i_c]
    max_tag_line_ch_ov,tag_L_ch_ov,lines_ch_ov,mesh_model = _create_lines(mesh_model,max_tag_line_ocean,p_list,False)

    p_list = [i_c,tag_right_c_l[0]]
    max_tag_line_ov,tag_L_ov,lines_L_ov,mesh_model        = _create_lines(mesh_model,max_tag_line_ch_ov,p_list,False)
    
    i_s = find_tag_line(coord_sub,c_phase.decoupling,'y')

    i_c = find_tag_line(coord_channel,c_phase.decoupling,'y')
    
    p_list = [i_s,i_c]
    max_tag_line,tag_base_ch,lines_base_ch,mesh_model         = _create_lines(mesh_model,max_tag_line_ov,p_list,False)
    if c_phase.cr !=0: 
        
        i_c = find_tag_line(coord_channel,c_phase.cr,'y')

        p_list = [i_c,tag_right_c_cr[0]]
        max_tag_line_crust,tag_L_cr,lines_cr,mesh_model       = _create_lines(mesh_model,max_tag_line,p_list,False)

        if c_phase.lc !=0:
            
            i_c = find_tag_line(coord_channel,(1-c_phase.lc)*c_phase.cr,'y')

            p_list = [i_c,tag_right_c_lcr[0]]
            max_tag_line_Lcrust,tag_L_Lcr,lines_lcr,mesh_model = _create_lines(mesh_model,max_tag_line_crust,p_list,False)

    line_global = np.hstack([lines_T,lines_R,lines_B,lines_L,lines_S,lines_ch,lines_oc,lines_ch_ov,lines_L_ov,lines_base_ch,lines_cr,lines_lcr])

    # Function create Physical line
    #-- Create Physical Line
    
    
    dict_tag_lines = {
        'Top'               : 1,
        'Right'             : 2,
        'Bottom'            : 3,
        'Left'              : 4,
        'Subduction'        : 5,
        'Channel'           : 6,
        'Oceanic'           : 7,
        'Channel_over'      : 8,
        'Overriding_mantle' : 9,
        'Channel_decoupling': 10,
        'Crust_overplate'   : 11,
        'LCrust_overplate'  : 12,
    }
 
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}
    
    mesh_model.addPhysicalGroup(1, tag_L_T, tag=dict_tag_lines['Top'])
    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, tag_L_R, tag=dict_tag_lines['Right'])


    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, tag_L_B, tag=dict_tag_lines['Bottom'])

    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, tag_L_L, tag=dict_tag_lines['Left'])

    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, tag_L_sub,tag=dict_tag_lines['Subduction'])

    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, tag_L_ch,tag=dict_tag_lines['Channel'])
    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, tag_L_oc, tag=dict_tag_lines['Oceanic'])
  
    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, tag_L_ch_ov, tag=dict_tag_lines['Channel_over']) 

    
    gmsh.model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, tag_L_ov, tag=dict_tag_lines['Overriding_mantle'])


    if c_phase.cr !=0: 

        mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

        mesh_model.addPhysicalGroup(1, tag_L_cr, tag=dict_tag_lines['Crust_overplate'])



        if c_phase.lc !=0:
            mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}
            mesh_model.addPhysicalGroup(1, tag_L_Lcr,tag=dict_tag_lines['LCrust_overplate'])
   

    # Left side of the subduction zone 10 
    # Oceanic crust 15
    # Right side of the subduction zone 20
    # Right mantle 25
    # Lithospheric mantle 25
    # Crust LC  30
    # Crust UC  35 
    # Channel 
    
    # left side of the subduction
    a = []
    a.extend(lines_oc[2,:])
    a.extend([lines_B[2,-1]])
    a.extend([-lines_L[2,0]])
    for i in range(len(a)):
        l = np.abs(a[i])
        i_l = np.where(line_global[2,:]==l)
        p0  = line_global[0,i_l][0][0]
        p1  = line_global[1,i_l][0][0]
        p_i_0 = np.where(global_points[2,:]==p0)
        p_i_1 = np.where(global_points[2,:]==p1)
        x   = [global_points[0,p_i_0][0],
               global_points[0,p_i_1][0]]
        y   = [global_points[1,p_i_0][0],
               global_points[1,p_i_1][0]]
        plt.plot(x,y,c='k')
    
    
    loop1 = mesh_model.geo.addCurveLoop(a,10) # Left side of the subudction zone

    # oceanic crust 
    a = []
    a.extend(lines_S[2,:])
    a.extend([lines_B[2,1]])
    b = (lines_oc[2,:])*-1
    a.extend(b[::-1])
    a.extend([-lines_L[2,-1]])
    for i in range(len(a)):
        l = np.abs(a[i])
        i_l = np.where(line_global[2,:]==l)
        p0  = line_global[0,i_l][0][0]
        p1  = line_global[1,i_l][0][0]
        p_i_0 = np.where(global_points[2,:]==p0)
        p_i_1 = np.where(global_points[2,:]==p1)
        x   = [global_points[0,p_i_0][0],
               global_points[0,p_i_1][0]]
        y   = [global_points[1,p_i_0][0],
               global_points[1,p_i_1][0]]
        plt.plot(x,y,c='b')
   
    
    
    
    loop2 = mesh_model.geo.addCurveLoop(a,15) # Oceanic crust 
    
    # right_mantle 
    # From p0 belonging ch -> ch->base channel -> subduction -> 
    
    a = []
    a.extend([-lines_R[2,-1]])
    a.extend([-lines_B[2,0]])
    # find index subduction 
    'Select the node of the slab from bottom to the base of decoupling'
    index = find_line_index(lines_S,coord_sub,c_phase.decoupling)
    index = index 
    buf_array = lines_S[2,index:]
    chose_sub = -buf_array 
    chose_sub = chose_sub[::-1]
    a.extend(chose_sub)
    a.extend(lines_base_ch[2,:])
    # Make a small function out of it. 
    
    index = find_line_index(lines_ch,coord_channel,c_phase.lt_d)
    index = index 
    choose_sub = []
    buf_array = np.array(lines_ch[2,index:])
    chose_sub = -1*buf_array 
    chose_sub = chose_sub[::-1]
    a.extend(chose_sub)
    a.extend(lines_L_ov[2,:])

    for i in range(len(a)):
        l = np.abs(a[i])
        i_l = np.where(line_global[2,:]==l)
        p0  = line_global[0,i_l][0][0]
        p1  = line_global[1,i_l][0][0]
        p_i_0 = np.where(global_points[2,:]==p0)
        p_i_1 = np.where(global_points[2,:]==p1)
        x   = [global_points[0,p_i_0][0],
               global_points[0,p_i_1][0]]
        y   = [global_points[1,p_i_0][0],
               global_points[1,p_i_1][0]]
        plt.plot(x,y,c='r')
    


    loop2 = mesh_model.geo.addCurveLoop(a,20) # Right mantle 

    if c_phase.cr !=0:
            
        index_a    = find_line_index(lines_ch,coord_channel,c_phase.cr)
        index_b    = find_line_index(lines_ch,coord_channel,c_phase.lt_d)-1
        buf_array = lines_ch[2,index_a:index_b+1]
        buf_array = -buf_array[::-1]
        a = []

        a.extend([-lines_R[2,2]])
        a.extend([-lines_L_ov[2,:].item()])
        a.extend(buf_array)
        a.extend([lines_cr[2,:][0]])


        
        for i in range(len(a)):
            l = np.abs(a[i])
            i_l = np.where(line_global[2,:]==l)
            p0  = line_global[0,i_l][0][0]
            p1  = line_global[1,i_l][0][0]
            p_i_0 = np.where(global_points[2,:]==p0)
            p_i_1 = np.where(global_points[2,:]==p1)
            x   = [global_points[0,p_i_0][0],
                   global_points[0,p_i_1][0]]
            y   = [global_points[1,p_i_0][0],
                   global_points[1,p_i_1][0]]
            plt.plot(x,y,c='g')
    
        # Permanent subcrustal mantle 
        mesh_model.geo.addCurveLoop(a,25) # Right mantle
        
        if c_phase.lc !=0:
            a = []
            index_a    = find_line_index(lines_ch,coord_channel,(1-c_phase.lc)*c_phase.cr)
            index_b    = find_line_index(lines_ch,coord_channel,c_phase.cr)-1

            buf_array = -lines_ch[2,index_a:index_b+1]
            buf_array = buf_array[::-1]
            
            a = []

            a.extend([-lines_R[2,1]])
            a.extend([-lines_cr[2,:].item()])
            a.extend(buf_array)
            a.extend([lines_lcr[2,:].item()])

            
            for i in range(len(a)):
                l = np.abs(a[i])
                i_l = np.where(line_global[2,:]==l)
                p0  = line_global[0,i_l][0][0]
                p1  = line_global[1,i_l][0][0]
                p_i_0 = np.where(global_points[2,:]==p0)
                p_i_1 = np.where(global_points[2,:]==p1)
                x   = [global_points[0,p_i_0][0],
                       global_points[0,p_i_1][0]]
                y   = [global_points[1,p_i_0][0],
                       global_points[1,p_i_1][0]]
                plt.plot(x,y,c='firebrick')
    


            mesh_model.geo.addCurveLoop(a,30)
 
            a = []
            index_a    = 0
            index_b    = find_line_index(lines_ch,coord_channel,(1-c_phase.lc)*c_phase.cr)-1

            buf_array = -lines_ch[2,index_a:index_b+1]
            buf_array = buf_array[::-1]
            
            a.extend([lines_R[2,0]])
            a.extend([-lines_lcr[2,:]])
            a.extend(buf_array)
            a.extend([lines_T[2,1]])

            
            for i in range(len(a)):
                l = np.abs(a[i])
                i_l = np.where(line_global[2,:]==l)
                p0  = line_global[0,i_l][0][0]
                p1  = line_global[1,i_l][0][0]
                p_i_0 = np.where(global_points[2,:]==p0)
                p_i_1 = np.where(global_points[2,:]==p1)
                x   = [global_points[0,p_i_0][0],
                       global_points[0,p_i_1][0]]
                y   = [global_points[1,p_i_0][0],
                       global_points[1,p_i_1][0]]
                plt.plot(x,y,c='r')

            
            mesh_model.geo.addCurveLoop(a,35)

    a = []
    index_a = find_line_index(lines_ch,coord_channel,c_phase.lt_d)
    index = find_line_index(lines_S,coord_sub,c_phase.lt_d)

    buf_array = -lines_S[2,0:index]
    buf_array = buf_array[::-1]
    a.extend(np.int32(lines_ch[2,0:index_a]))
    a.extend([-lines_ch_ov[2,:].item()])
    a.extend(buf_array)
    a.extend([lines_T[2,0]])
    mesh_model.geo.addCurveLoop(a,40)





    a = []
    index_a = find_line_index(lines_S,coord_sub,c_phase.lt_d)
    index = find_line_index(lines_S,coord_sub,c_phase.decoupling)
    buf_array = -lines_S[2,index_a:index]
    buf_array = buf_array[::-1]
    index_a = find_line_index(lines_ch,coord_channel,c_phase.lt_d)
    a.extend(np.int32(lines_ch[2,index_a:]))
    a.extend([-lines_base_ch[2,:].item()])
    a.extend(buf_array)
    a.extend([lines_ch_ov[2,:].item()])
    for i in range(len(a)):
        l = np.abs(a[i])
        i_l = np.where(line_global[2,:]==l)
        p0  = line_global[0,i_l][0][0]
        p1  = line_global[1,i_l][0][0]
        p_i_0 = np.where(global_points[2,:]==p0)
        p_i_1 = np.where(global_points[2,:]==p1)
        x   = [global_points[0,p_i_0][0],
               global_points[0,p_i_1][0]]
        y   = [global_points[1,p_i_0][0],
               global_points[1,p_i_1][0]]
        plt.plot(x,y,c='k')
    
    mesh_model.geo.addCurveLoop(a,45)


    # -- Create surfaces 
    
    
    # Left side of the subduction zone 10 
    # Oceanic crust 15
    # Right side of the subduction zone 20
    # Right mantle 25
    # Lithospheric mantle 25
    # Crust LC  30
    # Crust UC  35 
    # Channel 40
    
    Left_side_of_subduction_surf   = gmsh.model.geo.addPlaneSurface([10],tag=100) # Left side of the subudction zone
    Oceanic_Crust_surf             = gmsh.model.geo.addPlaneSurface([15],tag=150) # Left side of the subudction zone
    Right_side_of_subduction_surf  = gmsh.model.geo.addPlaneSurface([20],tag=200) # Right side of the subudction zone    
    Lithhospheric_Mantle_surf      = gmsh.model.geo.addPlaneSurface([25],tag=250) # Right mantle
    Crust_LC_surf                  = gmsh.model.geo.addPlaneSurface([30],tag=300) # Crust LC
    Crust_UC_surf                  = gmsh.model.geo.addPlaneSurface([35],tag=350) # Crust LC
    Channel_surf_A                 = gmsh.model.geo.addPlaneSurface([40],tag=400) # Channel
    Channel_surf_B                 = gmsh.model.geo.addPlaneSurface([45],tag=450) # Channel

    
    mesh_model.geo.synchronize()

    mesh_model.addPhysicalGroup(2, [Left_side_of_subduction_surf], tag=10000)
    mesh_model.addPhysicalGroup(2, [Oceanic_Crust_surf], tag=15000)
    mesh_model.addPhysicalGroup(2, [Right_side_of_subduction_surf], tag=20000)
    mesh_model.addPhysicalGroup(2, [Lithhospheric_Mantle_surf], tag=25000)
    mesh_model.addPhysicalGroup(2, [Crust_LC_surf], tag=30000)
    mesh_model.addPhysicalGroup(2, [Crust_UC_surf], tag=35000)
    mesh_model.addPhysicalGroup(2, [Channel_surf_A], tag=40000)   
    mesh_model.addPhysicalGroup(2, [Channel_surf_B], tag=45000)   

    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.geo.mesh.setAlgorithm(2, 40000, 3)
    mesh_model.geo.mesh.setAlgorithm(2, 45000, 3)
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}


    mesh_model.mesh.generate(2)
    mesh_model.mesh.setOrder(2)
    gmsh.write("experimental.msh")
    
    mesh, cell_markers, facet_markers = gmshio.read_from_msh("experimental.msh", MPI.COMM_WORLD, gdim=2)
    proc = 0 
    if proc == 0:
        # Read in mesh
        msh = meshio.read("experimental.msh")

        # Create and save one file for the mesh, and one file for the facets
        triangle_mesh = create_mesh(msh, "triangle6", prune_z=True)
        line_mesh = create_mesh(msh, "line3", prune_z=True)
        meshio.write("mesh.xdmf", triangle_mesh)
        meshio.write("mt.xdmf", line_mesh)
    MPI.COMM_WORLD.barrier()
    
    

    
    return mesh, cell_markers, facet_markers 


def create_mesh(mesh, cell_type, prune_z=False):
    # From the tutorials of dolfinx
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data.astype(np.int32)]})
    return out_mesh



def density_alt(p, T, rho_0):
    """
    UFL-compatible density function with temperature and pressure dependence.
    Arguments:
        option_rho: int (0=constant, 1=T-dependent, 2=T+P-dependent)
        p: Pressure (ufl.Expr or Function)
        T: Temperature (ufl.Expr or Function)
        rho_0: Reference density (ufl.Expr or float)
    """
    alpha_0 = 2.832e-5
    alpha_1 = 3.79e-8
    T0 = 273.15

    # Base T dependence (option 1 or 2)
    thermal_exp = - (alpha_0 * (T - T0) + 0.5 * alpha_1 * (T**2 - T0**2))
    rho_T = rho_0 * exp(thermal_exp)

    # Pressure dependence (option 2 only)
    Kb = (2 * 100e9 * (1 + 0.25)) / (3 * (1 - 0.25 * 2))  # Bulk modulus
    rho_TP = rho_T * exp(p / Kb)

    # Conditional branching
    return rho_TP  # fallback zero


def density(ph, P, T,sc):
    rho = ufl.conditional(ufl.eq(ph, 10000), 3300.0,
          ufl.conditional(ufl.eq(ph, 15000), 2900.0,
          ufl.conditional(ufl.eq(ph, 20000), 3300.0,
          ufl.conditional(ufl.eq(ph, 25000), 3250.0,
          ufl.conditional(ufl.eq(ph, 30000), 2800.0,
          ufl.conditional(ufl.eq(ph, 35000), 2700.0,
          ufl.conditional(ufl.eq(ph, 40000), 3250.0,
          ufl.conditional(ufl.eq(ph, 45000), 3250.0,
          0.0))))))))  # default to 0 if not matched

    return density_alt(P*sc.stress, T, rho) / sc.rho

def compute_lithostatic_pressure(pdb,BC,ctrl,IOCtrl,sc): 
    
        
    mesh, cell_tags, facet_tags = create_parallel_mesh(ctrl,IOCtrl)
    mesh.geometry.x[:] *= np.array([1.0/sc.L, 1.0/sc.L, 1.0/sc.L]) # Scale the mesh to the correct size

    x = ufl.SpatialCoordinate(mesh)
    
    # Define the function space for the lithostatic pressure
    ph = fem.functionspace(mesh, ("DG", 0))      # Material ID function space # Defined in the cell space {element wise} apperently there is a black magic that 
    # automatically does the interpolation of the function space, i.e. the function space is defined in the cell space, but it is automatically interpolated to the nodes                                    
    V = fem.functionspace(mesh, ("Lagrange", 2)) # Function space for the solution 
    
    
    
    # Define the material property field
    phase = fem.Function(ph) # Create a function to hold the phase information
    phase.x.name = "phase"
    phase = assign_phases(dict_surf, cell_tags,phase) # Assign phases using the cell tags and physical surfaces -> i.e. 10000 = Mantle ? is going to assign unique phase to each node? 
    
    # Define the lithostatic pressure function
    P    = fem.Function(V)
    P_n  = fem.Function(V) # ? {still blinding following the God}
    q    = ufl.TestFunction(V) # Trial function for the nonlinear solver
    T    = fem.Function(V) # Temperature function, can be set to a constant or variable later
    T.x.array[:] = 1200+273.15  # Set a constant temperature for the test function -> this will be the previous timestep temperature in the future 

    
    # Set the boundary conditions for this specific problem
    fdim = mesh.topology.dim - 1    
    top_facets   = facet_tags.find(1)
    top_dofs    = fem.locate_dofs_topological(V, mesh.topology.dim-1, top_facets)
    bc = fem.dirichletbc(0.0, top_dofs, V)

    
    g = fem.Constant(mesh, PETSc.ScalarType([0.0, -ctrl.g]))    
    F = ufl.dot(ufl.grad(P), ufl.grad(q)) * ufl.dx - ufl.dot(ufl.grad(q),density(phase,P,T,sc)*g) * ufl.dx

    problem = NonlinearProblem(F, P, bcs=[bc])

    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-9
    solver.report = True


    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "gmres"
    opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
    opts[f"{option_prefix}pc_type"] = "hypre"
    opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
    opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
    opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"
    ksp.setFromOptions()

    log.set_log_level(log.LogLevel.INFO)
    n, converged = solver.solve(P)
    
    
    pr = P.x.array
    pr = pr * sc.stress  # Scale the pressure to the correct units
    print(f"Scaled lithostatic pressure: {np.min(pr/1e9):.2f} GPa, {np.max(pr/1e9):.2f} GPa")
    
    rho = fem.Function(ph)
    v = ufl.TestFunction(ph)
    rho_trial = ufl.TrialFunction(ph)
    rho_expr = density(phase, P, T,sc)

    a = ufl.inner(rho_trial, v) * ufl.dx
    L = ufl.inner(rho_expr, v) * ufl.dx
    fem.petsc.LinearProblem(a, L, bcs=[], u=rho).solve()
    cell_density = rho.x.array*sc.rho  # Scale the density to the correct units
    print(f"Scaled density: {np.min(cell_density):.2f} kg/m^3, {np.max(cell_density):.2f} kg/m^3")
    assert (converged)
    print(f"Number of interations: {n:d}")
    
    
    



if __name__ == "__main__":
    
    pdb,BC_spec,ctrl,sc,IOCtrl = set_initial_()
    compute_lithostatic_pressure(pdb,BC_spec,ctrl,IOCtrl,sc)
    
    
    
    
    

    
    
    
    
