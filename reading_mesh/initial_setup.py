import numpy as np
from matplotlib import path
from solver_function.function_fem import field_fem
from solver_function.function_fem import element_type
from reading_mesh.Mesh_c import MESH,Computational_Data
from solver_function.bc import bc_class
from solver_function.solution_fem import Sol
from solver_function.bc import find_node_interface
from solver_function.bc import Interfaces


def _generate_initial_setup_temperature(M:MESH,CD:Computational_Data,BC:bc_class,S:Sol,ctrl,lhs,scal):
    from scipy.interpolate import griddata

    ind_left = find_node_interface(BC.bc_int,M.node_num,Interfaces.left)

    if lhs.van_keken == 1: 
        ini_space = lhs.z/scal.L
        ini_temp = lhs.LHS/scal.Temp
    else:
        data = np.load('databases/slab_k{0}_Cp{1}_rho{2}.npz'.format(ctrl.option_k,ctrl.option_Cp,ctrl.option_rho))
        temp1D   = data['temperature']
        ini_time = data['t']
        ini_space= data['z']
        # convert coordinate system 1D cooling plate script to coordinate system used here 
        ini_space = -ini_space/scal.L
        # which plate age do we want to use? 
        dummy,nt = np.shape(ini_time)
        for i in range(0,nt):
            if ini_time[0,i] == ctrl.slab_age:
                timestep = i
        ini_temp = temp1D[timestep,:]/scal.Temp


    S.scalars.T_old = griddata(ini_space, ini_temp, M.y[0:CD.NfemT], method='linear')
    S.scalars.T_old[M.y[0:CD.NfemT]<=np.min(ini_space)] = ctrl.Tmax






    # find lithosphere: 
    ind_lithosphere =  find_node_interface(BC.bc_int,M.node_num,Interfaces.OVPv0_int)
    ind_channel     =  find_node_interface(BC.bc_int,M.node_num,Interfaces.Channel_int)
    
    #find ind belonging to lithosphere: 
    x_channel = M.x[ind_channel-1]
    y_channel = M.y[ind_channel-1]
    x_lithosphere = M.x[ind_lithosphere-1]
    y_lithosphere = M.y[ind_lithosphere-1]
    ind_channel_s = np.argsort(x_channel)
    ind_lithosphere_s = np.argsort(x_lithosphere)
    x_channel = x_channel[ind_channel_s]
    y_channel = y_channel[ind_channel_s]
    x_lithosphere = x_lithosphere[ind_channel_s]
    y_lithosphere = y_lithosphere[ind_channel_s]
    min_x = np.min(x_channel)
    min_x_C = np.max(x_channel)
    max_x = np.max(x_lithosphere)
    min_y = np.min(y_channel) # I put -5000 to be sure that i select the right nodes
    max_y = np.max(y_channel)

    channel = np.zeros([len(x_channel),2],dtype=float)
    lithosphere = np.zeros([2,2],dtype=float)
    right = np.zeros([2,2],dtype=float)
    top = np.zeros([2,2],dtype=float)




    channel[:,0] = x_channel
    channel[:,1] = y_channel 
    lithosphere[:,0] = [min_x_C,max_x]
    lithosphere[:,1] = [min_y,min_y]
    right[:,0]       = [max_x,max_x]
    right[:,1]       = [min_y,max_y]
    top[:,0]         = [max_x,min_x]
    top[:,1]         = [max_y,max_y]
    
    lit = np.concatenate((channel,lithosphere,right,top))
    lit_reshaped = lit[:, :2]  # Ensure it has shape (N, 2)
    p = path.Path(lit_reshaped)
    X = np.zeros([len(M.x[0:CD.NfemT]),2],dtype=float)
    X[:,0] = M.x[0:CD.NfemT]
    X[:,1] = M.y[0:CD.NfemT]
    p.contains_points(X)
    ind = p.contains_points(X)

    S.scalars.T_old[ind] = ctrl.Ttop + (X[ind,1]-max_y) *(ctrl.Ttop- ctrl.Tmax )/(max_y-min_y)
    S.scalars.T_old[M.y[0:CD.NfemT]<min_y] = ctrl.Tmax






    return S 



def generate_phase_field(M,BC):
    """
    Function to generate the phase field for the initial setup. 
    The function uses the mesh and the computational data to generate the phase field. 
    The function also uses the boundary conditions and the solution object to generate the phase field. 
    The function returns the solution object with the phase field. 

    Parameters
    ----------
    M : MESH
        Mesh object.
    CD : Computational_Data
        Computational data object.
    BC : bc_class
        Boundary conditions object.
    S : Sol
        Solution object.
    ctrl : NumericalControls
        Numerical controls object.

    Returns
    -------
    S : Sol
        Solution object with the phase field.
    """

    #M = _fill_the_lithosphere(BC,M)
    M = _fill_the_channel(BC,M)

    return M


def _fill_the_lithosphere(BC,M):
    ind_lithosphere =  find_node_interface(BC.bc_int,M.node_num,Interfaces.OVPv0_int)
    ind_channel     =  find_node_interface(BC.bc_int,M.node_num,Interfaces.Channel_int)


    #find ind belonging to lithosphere: 
    x_channel = M.x[ind_channel-1]
    y_channel = M.y[ind_channel-1]
    x_lithosphere = M.x[ind_lithosphere-1]
    y_lithosphere = M.y[ind_lithosphere-1]
    ind_channel_s = np.argsort(x_channel)
    ind_lithosphere_s = np.argsort(x_lithosphere)
    x_channel = x_channel[ind_channel_s]
    y_channel = y_channel[ind_channel_s]
    x_lithosphere = x_lithosphere[ind_channel_s]
    y_lithosphere = y_lithosphere[ind_channel_s]
    min_x = np.min(x_channel)
    min_x_C = np.max(x_channel)
    max_x = np.max(x_lithosphere)
    min_y = np.min(y_channel) -500 # I put -5000 to be sure that i select the right nodes
    max_y = np.max(y_channel)

    channel = np.zeros([len(x_channel),2],dtype=float)
    lithosphere = np.zeros([2,2],dtype=float)
    right = np.zeros([2,2],dtype=float)
    top = np.zeros([2,2],dtype=float)




    channel[:,0] = x_channel
    channel[:,1] = y_channel 
    lithosphere[:,0] = [min_x_C,max_x]
    lithosphere[:,1] = [min_y,min_y]
    right[:,0]       = [max_x,max_x]
    right[:,1]       = [min_y,max_y]
    top[:,0]         = [max_x,min_x]
    top[:,1]         = [max_y,max_y]
    
    lit = np.concatenate((channel,lithosphere,right,top))
    lit_reshaped = lit[:, :2]  # Ensure it has shape (N, 2)
    p = path.Path(lit_reshaped)
    X = np.zeros([len(M.x),2],dtype=float)
    X[:,0] = M.x
    X[:,1] = M.y
    p.contains_points(X)
    ind = p.contains_points(X)
    M.Phase[ind] = 2
    
    return M

def _fill_the_channel(BC,M):
    
    ind_channel     =  find_node_interface(BC.bc_int,M.node_num,Interfaces.Channel_int)
    ind_subduction     =  find_node_interface(BC.bc_int,M.node_num,Interfaces.Subduction_int)

    y_subduction = M.y[ind_subduction-1]
    ind_s        = np.where(y_subduction >= np.min(M.y[ind_channel-1]))[0]
    ind_subduction = ind_subduction[ind_s]
    x_subduction = M.x[ind_subduction-1]  
    y_subduction = M.y[ind_subduction-1]
    x_channel = M.x[ind_channel-1]
    y_channel = M.y[ind_channel-1]

    ind_channel_s = np.argsort(x_channel)
    ind_subduction_s= np.argsort(x_subduction)

    x_channel = x_channel[ind_channel_s]
    y_channel = y_channel[ind_channel_s]
    x_subduction = x_subduction[ind_subduction_s]
    y_subduction = y_subduction[ind_subduction_s]
    
    min_x_C = np.min(x_channel)
    max_x_C = np.max(x_channel)
    min_x_S = np.min(x_subduction)
    max_x_S = np.max(x_subduction)
    min_y_S = np.min(y_subduction)
    max_y_S = np.max(y_subduction)
    
    
    channel = np.zeros([len(x_channel),2],dtype=float)
    subduction = np.zeros([len(x_subduction),2],dtype=float)
    bottom = np.zeros([2,2],dtype=float)
    top = np.zeros([2,2],dtype=float)




    channel[:,0] = x_channel
    channel[:,1] = y_channel 
    subduction[:,0] = x_subduction
    subduction[:,1] = y_subduction
    bottom[:,0]       = [max_x_S,max_x_C]
    bottom[:,1]       = [min_y_S,np.min(y_channel)]
    top[:,0]         = [min_x_C,min_x_S]
    top[:,1]         = [max_y_S,max_y_S]
    
    lit = np.concatenate((subduction,bottom,channel,top))
    lit_reshaped = lit[:, :2]  # Ensure it has shape (N, 2)
    p = path.Path(lit_reshaped)
    X = np.zeros([len(M.x),2],dtype=float)
    X[:,0] = M.x
    X[:,1] = M.y
    p.contains_points(X,radius=1e-5)
    ind = p.contains_points(X)
    M.Phase[ind] = 1



    return M