import sys 
import os

import numpy as np
import time as timing
from scipy.sparse import lil_matrix
import scipy.sparse as sps
import scipy.sparse.linalg.dsolve as linsolve
from scipy.special import erf
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpmath import mpf, mpc, mp
from reading_mesh.Mesh_c import MESH
from reading_mesh.Mesh_c import Computational_Data
from enum import Enum
from matplotlib import path
from numba.experimental import jitclass
from numba import float64, int32, boolean




class Interfaces(Enum):
    Subduction_int = 101 
    Channel_int    = 102 
    OVPv0_int      = 104
    OVPv1_int      = 105 
    top            = 111
    right          = 112 
    bottom         = 113 
    left           = 114

Dic_int ={'Subduction_surface': Interfaces.Subduction_int,
          'Channel_surface':      Interfaces.Channel_int,
          'OVP_v0'         :      Interfaces.OVPv0_int,
          'OVP_v1'         :      Interfaces.OVPv1_int,
          'top'            :      Interfaces.top,
          'right'          :      Interfaces.right,
          'bottom'         :      Interfaces.bottom,
          'left'           :      Interfaces.left}



spec = [
    ('bc_stokes_fix', boolean[:]),
    ('bc_stokes_val', float64[:]),
    ('bc_temp_fix', boolean[:]),
    ('bc_temp_val', float64[:]),
    ('bc_pressure_fix', boolean[:]),
    ('bc_pressure_val', float64[:]),
    ('bc_pressured_fix', int32[:]),
    ('bc_pressured_val', float64[:]),
    ('ind_P', int32),
    ('bc_int', int32[:,:])
]

@jitclass(spec)
class bc_class:
    def __init__(self, NfemV, NfemT, NfemP ,bc_int):
        self.bc_stokes_fix = np.zeros(NfemV, dtype=np.bool_)
        self.bc_stokes_val = np.zeros(NfemV, dtype=np.float64)
        self.bc_temp_fix = np.zeros(NfemT, dtype=np.bool_)
        self.bc_temp_val = np.zeros(NfemT, dtype=np.float64)
        self.bc_pressure_fix = np.zeros(NfemT, dtype=np.bool_)
        self.bc_pressure_val = np.zeros(NfemT, dtype=np.float64)
        self.bc_pressured_fix = np.zeros(NfemT*2, dtype=int32)
        self.bc_pressured_val = np.zeros(NfemT*2, dtype=np.float64)
        self.ind_P            = 0 

        self.bc_int = bc_int







# Boundary -> [0,1,2,3] => 0 = Top, 1 = Bottom, 2 = right, 3 = left
# top_type {0,1,2,3} => type 0 Do nothing type 1 No Slip type 2 Free Slip type 4 Traction 
# top_val  {val}     => [0,0] -> vector two component -> traction take directly lithostatic pressure computed with the algorithm of Dave May

#feature1  [] => code from gmesh -> identification number 
#feature_type {0,1,2,3}
# val [vec]

def _is_same_point(p1,p2):
    if (p1[0] == p2[0]) & (p1[1]==p2[1]):
        val = True
    else:
        val = False
    return val 



def _find_edge_element_int_surf(M,BC,int,el_bc,flag):

    ind_int = find_node_interface(BC.bc_int,M.node_num,int)
    vertex = [0,0,0]
    for iel in range(M.nel):
        x1, y1 = M.x[M.el_con[iel, 0] - 1], M.y[M.el_con[iel, 0] - 1]
        x2, y2 = M.x[M.el_con[iel, 1] - 1], M.y[M.el_con[iel, 1] - 1]
        x3, y3 = M.x[M.el_con[iel, 2] - 1], M.y[M.el_con[iel, 2] - 1]
        c = 0 
        for iint in range(len(ind_int)):
            x_int = M.x[ind_int[iint]-1]
            y_int = M.y[ind_int[iint]-1]
            if (_is_same_point([x1,y1],[x_int,y_int])) | (_is_same_point([x2,y2],[x_int,y_int])) | (_is_same_point([x3,y3],[x_int,y_int])):
                
                if (_is_same_point([x1,y1],[x_int,y_int])): 
                    c += 1 
                    vertex[0] = 1
                if (_is_same_point([x2,y2],[x_int,y_int])):
                    c += 1
                    vertex[1] = 1
                if (_is_same_point([x3,y3],[x_int,y_int])):
                    c += 1
                    vertex[2] = 1
                if c > 1:
                    el_bc[iel,0] = flag
                    if vertex[0] == 1 and vertex[1] == 1:
                       el_bc[iel,1] = 0
                       el_bc[iel,2] = 5
                       el_bc[iel,3] = 1
                    elif vertex[0] == 1 and vertex[2] == 1:
                       el_bc[iel,1] = 0
                       el_bc[iel,2] = 4
                       el_bc[iel,3] = 2
                    elif vertex[1] == 1 and vertex[2] == 1:
                       el_bc[iel,1] = 1
                       el_bc[iel,2] = 5
                       el_bc[iel,3] = 2
                break
                    



                   








#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def extract_node_internal_BC(LB:list,nod:int):
    """
    Function to give a flag to the boundary node.
    Input: 
    LB: List of boudary
    nod: num node
    Output: 
    bc_int: vector of integer-> 100-800 => physical line from gmsh
    """
    bc_int = np.zeros([len(LB)+1,nod+1],dtype=np.int32)  
    for i in reversed(range(len(LB))): 
        nodes_interface = LB[i][1]-1
        bc_int[i,nodes_interface]=Dic_int[LB[i][0]].value  
        bc_int[i,-1]=Dic_int[LB[i][0]].value
    # Find overlapping nodes
    for i in range(0,len(bc_int[0,:])):
        index = bc_int[:,i]
        index = index[index!=0]
        if len(index)>1:
            bc_int[8,i]=len(index)
        index=[]
    return bc_int
         



def set_boundary_condition_energy(M:MESH,BC:bc_class,u:float,v:float,ctrl,lhs,BCE,scal)->bc_class:
    """
    Input: 
    M: Mesh object -> contains the mesh information
    BC: bc_class -> contains the boundary condition
    u: float -> velocity field 
    input: input -> contains the input parameter [TO MODIFY]
    Output:
    BC: bc_class -> contains the boundary condition -> updated
    ========================================================================
    Function that set the boundary condition for the energy equation
    1. I need to divide into small functions, as usual, because, it appears that 
    I have a few problem between the arrays.     
    2. Iris created a connectivity matrix for the temperature: i think that I can 
    avoid this waste of memory: temperature mesh is basically a triangle of second order
    without central node. Since in the unstructured grid there is no god, and everything is 
    not following any kind of rule, all the central node are dumped at the end of the array
    Moreover, there is no risk that the central node are at the edge of the boundary conditions. 
    -> the only annoyance is that M.x/y are defined as full vector, and Python is pretty much 
    not obeying my implementation. To avoid to create an abomination, I preferred to
    divided into small functions. 
    """
    
    BC = left_BC_energy(M,BC,ctrl,lhs,BCE,scal)
    BC = right_BC_energy(M,BC,ctrl.Ttop,ctrl.Tmax,u,BCE,lhs)
    BC = top_BC_energy(M,BC,BCE)
    BC = bottom_BC_energy(M,BC,ctrl.Ttop,ctrl.Tmax,v,BCE)

    print('->Boundary condition set<-')
     
    return BC





def right_BC_energy(M:MESH,BC:bc_class,TTop:float,Tmax:float,u:float,BCE,lch):
    """
    Input: 
    M: Mesh object -> contains the mesh information
    BC: bc_class -> contains
    Tmax: float -> maximum temperature
    kelvin: float -> kelvin
    u: float -> velocity field 
    Output:
    BC: bc_class -> contains the boundary condition
    ========================================================================
    Function that set the right boundary condition for the energy equation
    All the nodes that are above the depth of the continental lithosphere are
    set to a linear profile {in the future I can introduce a few modification for
    having a continental geotherm}.
    All the nodes that are below the depth of the continental lithosphere are unconstrained
    -> unless u<0 {which implies that the flow is coming from outside}
    in this case the temperature is set to be the maximum temperature.

    Documentation: I tried to select in the following way the choosen nodes:
    BC.bc_temp_fix[y_r[ind_right-1]>depth_cont_lith] = True
    => This is wrong, because the boolean mask is not matching the BC.bc_temp_fix
    Why? Because I produced an array that is lesser than BC.bc_temp_fix. 
    -> When I do these tricks, python is expecting an array of the same size
    A=[0,1,2,3,4]
    C=[True,True,False,False,True]
    ->A[C] = x 
    if i give: 
    C=[True,True,True]
    -> it cannot verify the conditions per each of the element. 
    """ 
    ind_right = find_node_interface(BC.bc_int,M.node_num,Interfaces.right)
    if BCE.Rig_t == 0:
        true = True; val = BCE.right_v[0]
    elif BCE.Rig_t == 1: 
        true = False; val = 0.0 
    else: 
        true = False; val = 0.0 

    BC.bc_temp_fix[ind_right-1]=true; BC.bc_temp_val[ind_right-1]=val
    


    if BCE.Fea_on == 1:
        ind_ovp0 = find_node_interface(BC.bc_int,M.node_num,Interfaces.OVPv0_int)
        depth_cont_lith = lch.d_RHS  #np.min(M.y[ind_ovp0-1])
        y_r = M.y[0:(M.nv-M.nel)]


        # Ensure the boolean mask and BC.bc_temp_fix have matching dimensions
        valid_indices = (ind_right - 1)[y_r[ind_right - 1] >= depth_cont_lith]
        valid_indices2 = (ind_right - 1)[(u[ind_right - 1] < 0.0) & (y_r[ind_right - 1] <= depth_cont_lith)]
        T_right = TTop + (y_r[valid_indices]-np.max(y_r)) *(TTop - Tmax )/(np.max(y_r)-depth_cont_lith)
        BC.bc_temp_fix[valid_indices] = True
        BC.bc_temp_val[valid_indices]=T_right
        BC.bc_temp_fix[valid_indices2]=True
        BC.bc_temp_val[valid_indices2]=Tmax 

    return BC 

def bottom_BC_energy(M:MESH,BC:bc_class,TTop:float,Tmax:float,v:float,BCE):
    """
    Input: 
    M: Mesh object -> contains the mesh information
    BC: bc_class -> contains
    Tmax: float -> maximum temperature
    kelvin: float -> kelvin
    u: float -> velocity field 
    Output:
    BC: bc_class -> contains the boundary condition
    ========================================================================
    Function that set the right boundary condition for the energy equation
    All the nodes that are above the depth of the continental lithosphere are
    set to a linear profile {in the future I can introduce a few modification for
    having a continental geotherm}.
    All the nodes that are below the depth of the continental lithosphere are unconstrained
    -> unless u<0 {which implies that the flow is coming from outside}
    in this case the temperature is set to be the maximum temperature.

    Documentation: I tried to select in the following way the choosen nodes:
    BC.bc_temp_fix[y_r[ind_right-1]>depth_cont_lith] = True
    => This is wrong, because the boolean mask is not matching the BC.bc_temp_fix
    Why? Because I produced an array that is lesser than BC.bc_temp_fix. 
    -> When I do these tricks, python is expecting an array of the same size
    A=[0,1,2,3,4]
    C=[True,True,False,False,True]
    ->A[C] = x 
    if i give: 
    C=[True,True,True]
    -> it cannot verify the conditions per each of the element. 
    """ 
    ind_bottom = find_node_interface(BC.bc_int,M.node_num,Interfaces.bottom)

    if BCE.Bot_t == 0:
        true = True; val = BCE.Bot_v[0]
    elif BCE.Bot_t == 1: 
        true = False; val = 0.0 
    elif BCE.Bot_t == 3:
        xm = np.min(M.x)
        xM = np.max(M.x)
        x_m = xm/2+xM/2
        true = True
        val = lambda x : BCE.Bot_v[0]+0.6*BCE.Bot_v[0]*np.exp(-((x-x_m)/0.25)**2) 
        BC.bc_temp_fix[ind_bottom-1]=true; BC.bc_temp_val[ind_bottom-1]=val(M.x[ind_bottom-1])
        return BC 

    else: 
        true = False; val = 0.0 

    BC.bc_temp_fix[ind_bottom-1]=true; BC.bc_temp_val[ind_bottom-1]=val

    if BCE.Fea_on == 1:
        valid_indices2 = (ind_bottom - 1)[v[ind_bottom - 1] > 0.0]
        BC.bc_temp_fix[valid_indices2]=True
        BC.bc_temp_val[valid_indices2]=Tmax
#
    return BC 


def top_BC_energy(M:MESH,BC:bc_class,BCE):
    """
    Input: 
    M: Mesh object -> contains the mesh information
    BC: bc_class -> contains  
    Output:
    BC: bc_class -> contains the boundary condition updated
    ========================================================================
    Function that set the top boundary condition for the energy equation; simple concept
    selct the boudnary node at the top.   
    """
    if BCE.Top_t == 0:
        true = True; val = BCE.Top_v[0]
    elif BCE.Top_t == 1: 
        true = False; val = 0.0 
    else: 
        true = False; val = 0.0 

    ind_top = find_node_interface(BC.bc_int,M.node_num,Interfaces.top)
    BC.bc_temp_fix[ind_top-1]=true; BC.bc_temp_val[ind_top-1]= val

    return BC 

def left_BC_energy(M:MESH,BC:bc_class,ctrl,lhs,BCE,scal):
    """
    Input: 
    M: Mesh object -> contains the mesh information
    BC: bc_class -> contains
    Tmax: float -> maximum temperature
    kelvin: float -> kelvin
    Output:
    BC: bc_class -> contains the boundary condition
    ========================================================================
    Function that set the left boundary condition for the energy equation
    Load the data from the database, select the proper temperature profile as a 
    function of the slab age, interpolate the data to the left boundary grid node. 
    -> Select the node that are below the depth of the lithosphere, and generate 
    an adiabatic profile of temperature. 
    -> return the BC update class. 

        """

    if BCE.Lef_t == 0:
        true = True; val = BCE.Top_v[0]
    elif BCE.Lef_t == 1: 
        true = False; val = 0.0 
    else: 
        true = False; val = 0.0 

    ind_left = find_node_interface(BC.bc_int,M.node_num,Interfaces.left)

    BC.bc_temp_fix[ind_left-1]=true; BC.bc_temp_val[ind_left-1]= val

    if BCE.Fea_on == 1:


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



        temp_LHS = griddata(ini_space, ini_temp, M.y[ind_left-1], method='linear')
        temp_LHS[M.y[ind_left-1]<=np.min(ini_space)] = ctrl.Tmax

        BC.bc_temp_fix[ind_left-1]=True; BC.bc_temp_val[ind_left-1]= temp_LHS


    return BC 


def set_boundary_condition_stokes(M:MESH,CD:Computational_Data,BC:bc_class,v:float,BCS,scal):

    nx,ny     = compute_normal(M,BC.bc_int)
    BC        = fill_boundary_condition_stokes(M,CD,BC,nx,ny,v,BCS,scal)
    print('->Boundary condition set<-')

    return BC 

def compute_normal(M:MESH,bc_int:int):
    """
    ===== OFFCOURSE the number of the node within the physical line seems to be generated by 
    a person under the influence of some substance. 
    ---- The assumption that the slab is a strict function of x AND that x & y are not repeating 
    -1) Select node 
    -2) Select coordinate
    -3) order the coordinate of the node
    -4) compute the vector dot product of the convergence velocity with the tangent vector/norm
    ========================================================================
    Input:
    M: Mesh object
    bc_int: vector of integer
    Output:
    nx,ny: vector of normal vector
    ========================================================================
    Function that extract the normal vector at the interface 
    NB: the curvilinear slab is constructed as set of linear segment: this entails 
    that the slope of the slab within this segment is constant. Since the velocity 
    is v [1,0] the velocity along the interface is u_s = tan_v*v + nor_v*n. Where tan_v and norm_v
    are the versor of the tangent and normal vector of a given segment. 

    """

    nx=np.zeros(M.nv,dtype=np.float64) # x component of the normal vector
    ny=np.zeros(M.nv,dtype=np.float64) # y component of the normal vector
    
    i_i = find_node_interface(bc_int,M.node_num,Interfaces.Subduction_int)
    
    count_av = np.zeros(M.nv,dtype=int) # count the number of time the node is used
    vc = [1,0]
    # Sort the coordinate of the node 
    X_s = M.x[i_i-1]
    Y_s = M.y[i_i-1]
    u, c = np.unique(X_s, return_counts=True)
    dup = u[c > 1]
    if len(dup)>0:
        raise ValueError('X coordinate of the node is not unique.The local profile of the slab must have monotonic x coordinate')

    i_s = np.argsort(X_s)
    i_i = i_i[i_s]
    # Compute the normal vector
    for i in range(len(i_i)-1):
    

        i0 = i_i[i]-1
        i1 = i_i[i+1]-1
        
        count_av[i0] +=1 
        count_av[i1] +=1 

        x0 = M.x[i0]
        x1 = M.x[i1]
        y0 = M.y[i0]
        y1 = M.y[i1]
        cy  = abs(y1-y0)
        cx  = abs(x1-x0)
      

        slope = np.arctan(cy/cx)
        
        v_nor = [-np.sin(slope),np.cos(slope)]
        v_tan = [np.cos(slope),-np.sin(slope)]
        u_s = vc[0]*v_tan+vc[1]*v_nor

        nx[i0] +=u_s[0]
        nx[i1] +=u_s[0] 
        ny[i0] +=u_s[1] 
        ny[i1] +=u_s[1]
     
    # Normalise the line with the amount of counts
    nx[nx!=0] = nx[nx!=0]/count_av[nx!=0]
    ny[nx!=0] = ny[nx!=0]/count_av[nx!=0]
    i_c = find_node_interface(bc_int,M.node_num,Interfaces.OVPv1_int)
    count_av2 = np.zeros(M.nv,dtype=int) # count the number of time the node is used

    X_s = M.x[i_c-1]
    i_sc = np.argsort(X_s)
    i_c = i_c[i_sc]
    for i in range(len(i_c)-1):
        i0 = i_c[i]-1
        i1 = i_c[i+1]-1
        x0 = M.x[i0]
        x1 = M.x[i1]
        y0 = M.y[i0]
        y1 = M.y[i1]
        cy  = abs(y1-y0)
        cx  = abs(x1-x0)

        slope = np.arctan(cy/cx)
        v_nor = [-np.sin(slope),np.cos(slope)]
        v_tan = [np.cos(slope),-np.sin(slope)]
        u_s = vc[0]*v_tan+vc[1]*v_nor
            
        # Here I do not understand: Iris imposes the velocity on the bottom as function of the normal and tangent vectors of the overriding plate. However, later in the BC she seems to 
        # impose a couette style velocity profile. In this case, the velocity vector must be parallel to the interface velocity field. In case I will restore her vision, for now
        # I will impose the velocity // to the one imposed by the slab
        if (nx[i0]!=0) & (i==0):
            a = nx[i0]
            b = ny[i0]
        else: 
            count_av2[i0] +=1 
            count_av2[i1] +=1 
            nx[i0] += a
            nx[i1] += a 
            ny[i0] += b 
            ny[i1] += b
        
    
    nx[count_av2!=0]= nx[count_av2!=0]/count_av2[count_av2!=0]    
    ny[count_av2!=0]= ny[count_av2!=0]/count_av2[count_av2!=0]    


    
    return nx,ny 


def fill_boundary_condition_stokes(M:MESH,CD:Computational_Data,BC:bc_class,nx:float,ny:float,v:float,BCS,scal):
    """
    Input: 
    M: Mesh object -> contains the mesh information
    BC: bc_class -> contains the boundary condition
    bc_int: vector of integer -> contains the boundary condition
    nx,ny: vector of float -> contains the normal vector
    Output:
    BC: bc_class -> contains the boundary condition
    ========================================================================
    Function that fill the boundary condition for the stokes equation
    ========================================================================
    Here there is a major problem: Iris in her previous code imposed the velocity 
    on the bottom, right (below the continental lithosphere) and left boundary. 
    The velocity is imposed using an analytical (?) solution for the corner flow. 
    I can port this function and assign the velocities, but I find an alternative solution
    that can gives more spontaneous results. The first implementation is the one 
    that follows xFieldstone, then I will modify the script such that the code can host 
    a nested boundary condition (i.e., the target domain is ebbended in a larger domain)

    """
    # Find the relevant index
    ind_top = find_node_interface(BC.bc_int,M.node_num,Interfaces.top)
    ind_right = find_node_interface(BC.bc_int,M.node_num,Interfaces.right)
    ind_bottom = find_node_interface(BC.bc_int,M.node_num,Interfaces.bottom)
    ind_left = find_node_interface(BC.bc_int,M.node_num,Interfaces.left)

    # Set the top boundary condition
    if BCS.Top_t == 0 or BCS.Top_t == 3:
        vx_true = False ; vy_true = False
        vx_val  = 0.0   ; vy_val  = 0.0
    elif BCS.Top_t == 1:
        vx_true = True ; vy_true = True
        vx_val  = 0.0   ; vy_val  = 0.0
    elif BCS.Top_t == 2:
        vx_true = False ; vy_true = True
        vx_val  = 0.0   ; vy_val  = 0.0

    BC.bc_stokes_fix[(ind_top-1)*CD.ndofV]   = vx_true ; BC.bc_stokes_val[(ind_top-1)*CD.ndofV]     = vx_val
    BC.bc_stokes_fix[(ind_top-1)*CD.ndofV+1] = vy_true ; BC.bc_stokes_val[(ind_top-1)*CD.ndofV+1]   = vy_val

    # Set the top boundary condition
    if BCS.Bot_t == 0 or BCS.Bot_t == 3:
        vx_true = False ; vy_true = False
        vx_val  = 0.0   ; vy_val  = 0.0
    elif BCS.Bot_t == 1:
        vx_true = True ; vy_true = True
        vx_val  = 0.0   ; vy_val  = 0.0
    elif BCS.Bot_t == 2:
        vx_true = False ; vy_true = True
        vx_val  = 0.0   ; vy_val  = 0.0


    BC.bc_stokes_fix[(ind_bottom-1)*CD.ndofV]   = vx_true ; BC.bc_stokes_val[(ind_bottom-1)*CD.ndofV]   = vx_val
    BC.bc_stokes_fix[(ind_bottom-1)*CD.ndofV+1] = vy_true ; BC.bc_stokes_val[(ind_bottom-1)*CD.ndofV+1]   = vy_val

        # Set the top boundary condition
    if BCS.Lef_t == 0 or BCS.Lef_t == 3:
        vx_true = False ; vy_true = False
        vx_val  = 0.0   ; vy_val  = 0.0
    elif BCS.Lef_t == 1:
        vx_true = True ; vy_true = True
        vx_val  = 0.0   ; vy_val  = 0.0
    elif BCS.Lef_t == 2:
        vx_true = True ; vy_true = False
        vx_val  = 0.0   ; vy_val  = 0.0

    BC.bc_stokes_fix[(ind_left-1)*CD.ndofV]   = vx_true ; BC.bc_stokes_val[(ind_left-1)*CD.ndofV]   = vx_val
    BC.bc_stokes_fix[(ind_left-1)*CD.ndofV+1] = vy_true ; BC.bc_stokes_val[(ind_left-1)*CD.ndofV+1]   = vy_val

    if BCS.Rig_t == 0 or BCS.Rig_t == 3:
        vx_true = False ; vy_true = False
        vx_val  = 0.0   ; vy_val  = 0.0
    elif BCS.Rig_t == 1:
        vx_true = True ; vy_true = True
        vx_val  = 0.0   ; vy_val  = 0.0
    elif BCS.Rig_t == 2:
        vx_true = True ; vy_true = False
        vx_val  = 0.0   ; vy_val  = 0.0

    BC.bc_stokes_fix[(ind_right-1)*CD.ndofV]   = vx_true ; BC.bc_stokes_val[(ind_right-1)*CD.ndofV]   = vx_val
    BC.bc_stokes_fix[(ind_right-1)*CD.ndofV+1] = vy_true ; BC.bc_stokes_val[(ind_right-1)*CD.ndofV+1]   = vy_val


    BC.bc_pressure_fix[(ind_top-1)]   = True ; BC.bc_pressure_val[(ind_top-1)]   = 0.0   # Lithostatic pressure BC


    if BCS.Fea_on ==1: 
        ind_subduction = find_node_interface(BC.bc_int,M.node_num,Interfaces.Subduction_int)
        ind_channel = find_node_interface(BC.bc_int,M.node_num,Interfaces.Channel_int)
        ind_ovp0 = find_node_interface(BC.bc_int,M.node_num,Interfaces.OVPv0_int)
        ind_ovp1 = find_node_interface(BC.bc_int,M.node_num,Interfaces.OVPv1_int)


       # Set the right boundary condition
        depth_cont_lith = np.max(M.y[ind_ovp0-1])
        ind_right = ind_right[M.y[ind_right-1]>depth_cont_lith]
        BC.bc_stokes_fix[(ind_right-1)*CD.ndofV]   = True ; BC.bc_stokes_val[(ind_right-1)*CD.ndofV]   = 0.0
        BC.bc_stokes_fix[(ind_right-1)*CD.ndofV+1] = True ; BC.bc_stokes_val[(ind_right-1)*CD.ndofV+1]   = 0.0
        # Set the bottom PLATE boundary condition
        BC.bc_stokes_fix[(ind_ovp0-1)*CD.ndofV]   = True ; BC.bc_stokes_val[(ind_ovp0-1)*CD.ndofV]   = 0.0
        BC.bc_stokes_fix[(ind_ovp0-1)*CD.ndofV+1] = True ; BC.bc_stokes_val[(ind_ovp0-1)*CD.ndofV+1]   = 0.0
        # Set the CHANNEL boundary condition
        ind_c = (ind_channel - 1)[ (M.y[ind_channel - 1] >= (depth_cont_lith))]

        BC.bc_stokes_fix[(ind_c)*CD.ndofV]   = True ; BC.bc_stokes_val[(ind_c)*CD.ndofV]   = 0.0  
        BC.bc_stokes_fix[(ind_c)*CD.ndofV+1] = True ; BC.bc_stokes_val[(ind_c)*CD.ndofV+1]   = 0.0
        #set Subduction boundary cond
        ind_s = (ind_subduction - 1)[ (M.y[ind_subduction - 1] >= (np.min(M.y)-0e3/660e3))]

        BC.bc_stokes_fix[(ind_s)*CD.ndofV]   = True ; BC.bc_stokes_val[(ind_s)*CD.ndofV]   = nx[ind_s-1]*v
        BC.bc_stokes_fix[(ind_s)*CD.ndofV+1] = True ; BC.bc_stokes_val[(ind_s)*CD.ndofV+1]   = ny[ind_s-1]*v
        # Set the inside channel boundary condition
        inda = np.where(M.y[ind_subduction-1]==depth_cont_lith)
        indb = np.where(M.y[ind_channel-1]==depth_cont_lith)
        x0 = M.x[ind_subduction[inda]-1]
        x1 = M.x[ind_channel[indb]-1]
        scale = 1-1/(x1-x0)*(M.x-x0)
        ind_node = M.node_num[(M.x>=x0) & (M.x<=x1) & (M.y==depth_cont_lith)]
        BC.bc_stokes_fix[(ind_node-1)*CD.ndofV]   = True ; BC.bc_stokes_val[(ind_node-1)*CD.ndofV]   = nx[ind_node-1]*scale[ind_node-1]*v*0.0
        BC.bc_stokes_fix[(ind_node-1)*CD.ndofV+1] = True ; BC.bc_stokes_val[(ind_node-1)*CD.ndofV+1]   = ny[ind_node-1]*scale[ind_node-1]*v*0.0
        for i in range(0,M.nv):
            if  M.x[i]<1e3/660e3 and M.y[i]>-0.5/660e3 and M.x[i]>0:
              BC.bc_stokes_fix[i*CD.ndofV  ] = False 
              BC.bc_stokes_fix[i*CD.ndofV+1] = False 




    return BC 






def find_node_interface(bc_int:int,nod_list:int,interface:Interfaces):
    ind_bound = np.where(bc_int[:,-1]==interface.value)[0]
    i_list_B = bc_int[ind_bound[0],:]
    i_i = nod_list[i_list_B[:-1]==interface.value] # node number of the interface
    return i_i

# Place holder for the velocity boundary condition


def compute_corner_flow_velocity(x,y,l1,l2,l3,angle,v0,Lx,Ly):
    """
    Fuction that compute the velocity field for the corner flow, 
    coming directly from the xFieldstone code.
    Input:
    x,y: coordinates
    
    """


    v1=-v0
    theta0=angle
    theta1=np.pi-theta0
    l4=l3*np.tan(theta0)
    A0 = (- theta0 * np.sin(theta0))/(theta0**2-np.sin(theta0)**2 ) *v0 
    B0=0
    C0=(np.sin(theta0)-theta0*np.cos(theta0))/(theta0**2-np.sin(theta0)**2 ) * v0
    D0=-A0
    A1 =1./(theta1**2-np.sin(theta1)**2 ) * \
        ( -v0*theta1*np.sin(theta1)-v1*theta1*np.cos(theta1)*(np.sin(theta1)+theta1*np.cos(theta1))\
        +v1*(np.cos(theta1)-theta1*np.sin(theta1))*theta1*np.sin(theta1) )   
    B1=0
    C1=1./(theta1**2-np.sin(theta1)**2 ) * \
       ( v0*(np.sin(theta1)-theta1*np.cos(theta1)) + v1*theta1**2*np.cos(theta1)*np.sin(theta1) \
       - v1*(np.cos(theta1)-theta1*np.sin(theta1))*(np.sin(theta1)-theta1*np.cos(theta1)) )   
    D1=-A1

    u=0.
    v=0.

    #------------------------
    # slab left 
    #------------------------
    if y>=Ly-l1 and x<=l3:
       u=v0
       v=0.

    #------------------------
    # slab 
    #------------------------
    if x>=l3 and y<=Ly+l4-x*np.tan(theta0) and y>=Ly+l4-x*np.tan(theta0)-l1:
       u=v0*np.cos(theta0)
       v=-v0*np.sin(theta0)

    #------------------------
    # overriding plate
    #------------------------
    if y>Ly+l4-x*np.tan(theta0) and y>Ly-l2:
       u=0.0
       v=0.0

    #------------------------
    # wedge
    #------------------------
    xC=l3+l2/np.tan(theta0)
    yC=Ly-l2
    if x>xC and y<yC:
       xt=x-xC 
       yt=yC-y 
       theta=np.arctan(yt/xt) 
       r=np.sqrt((xt)**2+(yt)**2)
       if theta<theta0:
          # u_r=f'(theta)
          ur = A0*np.cos(theta)-B0*np.sin(theta) +\
               C0* (np.sin(theta)+theta*np.cos(theta)) + D0 * (np.cos(theta)-theta*np.sin(theta))
          # u_theta=-f(theta)
          utheta=- ( A0*np.sin(theta) + B0*np.cos(theta) + C0*theta*np.sin(theta) + D0*theta*np.cos(theta))
          ur=-ur
          utheta=-utheta
          u=  ur*np.cos(theta)-utheta*np.sin(theta)
          v=-(ur*np.sin(theta)+utheta*np.cos(theta)) # because of reverse orientation

    #------------------------
    # under subducting plate
    #------------------------
    xD=l3
    yD=Ly-l1
    if y<yD and y<Ly+l4-x*np.tan(theta0)-l1:
       xt=xD-x 
       yt=yD-y 
       theta=np.arctan2(yt,xt) #!; write(6548,*) theta/pi*180
       r=np.sqrt((xt)**2+(yt)**2)
       #u_r=f'(theta)
       ur = A1*np.cos(theta) - B1*np.sin(theta) + C1* (np.sin(theta)+theta*np.cos(theta)) \
            + (D1-v1) * (np.cos(theta)-theta*np.sin(theta))
       #u_theta=-f(theta)
       utheta=- ( A1*np.sin(theta) + B1*np.cos(theta) + C1*theta*np.sin(theta) + (D1-v1)*theta*np.cos(theta))
       ur=-ur
       utheta=-utheta
       u=-(ur*np.cos(theta)-utheta*np.sin(theta))
       v=-(ur*np.sin(theta)+utheta*np.cos(theta)) #! because of reverse orientation

    return u,v










