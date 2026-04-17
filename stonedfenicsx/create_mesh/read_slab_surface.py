from stonedfenicsx.package_import import * 

from scipy.optimize import curve_fit

import numpy as np

from scipy.special import expit

from scipy.interpolate import CubicSpline

from stonedfenicsx.create_mesh.aux_create_mesh import create_slab_surface


def apply_Savitzky_Golay(yd,order,cf):
    
    from scipy.signal import savgol_filter as sgf 
    
    yd_smooth = sgf(yd,cf,order)
    
    return yd_smooth

def function_bending_data(ell:NDArray[float],theta:NDArray[float])->float:
    
    cs=CubicSpline(ell,theta)
    
    
    def f2(x:float| NDArray[float]):
        
        return cs(x)
    

    return f2  

def curve_fitting(xd:NDArray[float],yd:NDArray[float],path:str , name:str)->NDArray[float]:
    """
    """
        

    
    
    
    theta, ell, xd,yd = compute_bending_angle_length(xd,yd)

    theta_sgf = apply_Savitzky_Golay(theta,2,100)        
    
    f_theta = function_bending_data(ell,theta_sgf)

    if np.min(yd) > -500: 
        mind = np.min(yd)
    else: 
        mind = -500 

    slab_top,theta_mean,ell_s = create_slab_surface(f_theta,mind,stp = 5.0)
  


    
    
      
  
  
#  
#    fig = plt.figure()
#    ax = fig.gca()
#    ax.plot(ell,theta*180/np.pi,c='k',alpha=0.1,label=r'$\theta$ original')
#    ax.plot(ell,theta_sgf*180/np.pi, c='b', label = r'$\theta$ filtered data')
#    ax.plot(ell_s,theta_mean*180/np.pi,c='r',label=r'$\theta$  post processed')
#    ax.set_xlabel(r'$\ell$ [km]')
#    ax.set_ylabel(r'$\vartheta$ [deg]')
#    # Set spine thickness
#    for spine in ax.spines.values():
#        spine.set_linewidth(1.2)
#
#    # Set spine color
#    for spine in ax.spines.values():
#        spine.set_color("k")
#
#    ax.grid(visible=True,axis='both',which='both',color='k',linestyle=':',linewidth=0.5)
#    plt.legend()
#    fig.savefig(f'{path}/{name}_bending_angle.png',dpi=600)
#    
#    fig = plt.figure()
#    ax = fig.gca()
#    ax.plot(xd,yd,c='k',alpha=0.1,label='original')
#    ax.plot(slab_top[:,0],slab_top[:,1],c='r',label='post processed')
#    # Set spine thickness
#    for spine in ax.spines.values():
#        spine.set_linewidth(1.2)
#
#    # Set spine color
#    for spine in ax.spines.values():
#        spine.set_color("k")
#
#    ax.set_xlabel(r'Distance [km]')
#    ax.set_ylabel(r'Depth [km]')
#
#    ax.grid(visible=True,axis='both',which='both',color='k',linestyle=':',linewidth=0.5)
#    plt.legend()
#    fig.savefig(f'{path}/{name}_surface.png',dpi=600)
#    
#    
#    
#
#    
#


    return slab_top,theta_mean




def compute_bending_angle_length(x:NDArray[float],z:NDArray[float])->tuple[NDArray[float],NDArray[float]]:
    """From the given subduction surface compute the bending angle and the total length
    of the slab

    Args:
        x (NDArray[float]): coordinate x of the top surface of the slab
        z (NDArray[float]): coordinate z of the top surface of the slab

    Returns:
        theta: bending angle of the top surface of the slab
        ell : total length of the surface of the slab. 
    """
    
    def cx(x0:float,x1:float)->float: 
        """Correct coordinate with the first point x/y coordinate

        Args:
            x0 (float): coordinate x or y of the first point 
            x1 (float): coordinate x or y of the second point

        Returns:
            float: dx/dy of the given couple of point
        """
        return x1-x0

    def distance(x0:NDArray[float],x1:NDArray[float])->float:
        """Compute the distance between two points 

        Args:
            x0 (NDArray[float]): point 1 [x,y] coordinate
            x1 (NDArray[float]): point 2 [x,y] coordinate

        Returns:
            float: distance between the two points 
        """
        return np.sqrt(cx(x0[0],x1[0])**2+cx(x0[1],x1[1])**2)

    pmax = len(x)-1 
    
    theta = np.zeros(pmax+1,dtype=np.float64)
    
    ell = np.zeros(pmax+1,dtype=np.float64)
    
    for i in range(pmax): 
        x0 = [x[i],z[i]]
        x1 = [x[i+1],z[i+1]]
        dx = cx(x0[0],x1[0])
        d  = distance(x0,x1)
        theta[i] = dx/d
        if i == pmax-1: 
            theta[i+1] = theta[i]  
        elif i ==0:
            # Compute the distance of the slab assuming that theta is constant and equal to the first segment of the slab           
            dy = cx(x0[1],x1[1])
            x_t = -x0[1]*(dx/dy)+x0[0]
            sx = cx(x_t,x0[0])
            dl = distance([x_t,0.0],x0)
            theta_0 = sx/dl
            ell[i] = dl            
        
        ell[i+1] = ell[i]+d 

    # Adding an additional portion of the vector 
    # Create a tail vector: 
    imax = np.int32(np.floor(dl/np.mean(np.diff(ell)))) 
    dl_tail = dl/(imax)
    
    theta_tail = np.zeros(imax)
    ell_tail = np.zeros(imax)
    x_tail = np.zeros(imax)
    y_tail = np.zeros(imax)

    theta = np.acos(theta)
    theta_0 = np.acos(theta_0)

    for i in range(imax): 
        
        if i == 0: 
            theta_tail[i] = theta_0 
            ell_tail[i] = 0.0 
            x_tail[i] = x_t
            y_tail[i] = 0.0 
        else:
            ell_tail[i] = ell_tail[i-1]+dl_tail
        if i < imax-1:
            theta_tail[i+1] = theta_0
            x_tail[i+1] = x_tail[i] + dl_tail*(np.cos(theta_tail[i]))
            y_tail[i+1] = y_tail[i] - dl_tail*(np.sin(theta_tail[i]))
        
    
        

    theta = np.concatenate((theta_tail,theta))
    ell = np.concatenate((ell_tail,ell))
    x = np.concatenate((x_tail,x))
    z = np.concatenate((y_tail,z))


    
    return theta, ell, x-x_t, z 


def read_file_slab(file_path:str)->tuple[NDArray[float],NDArray[float]]:
    
    from pathlib import Path
    
    slab= np.loadtxt(file_path)
    ax = slab[:,0]
    ay = -slab[:,1]
    dz = slab[:,2]
    
    ax = ax[ay<-10]
    dz = dz[ay<-10]
    ay = ay[ay<-10]
    
    path = Path(file_path)

    slab_top, theta_mean = curve_fitting(ax,ay,path.parent,path.stem)


    return slab_top*1000, theta_mean 




def generate_parallel_layer_subducting_plate(sx:NDArray[np.float64],
                                      sy:NDArray[np.float64],
                                      th:NDArray[np.float64],
                                      lt:float)->tuple[NDArray[np.float64],NDArray[np.float64]]:
    """
    Compute the coordinates of an internal layer surface within the subducting plate.
    
    Given the slab top surface coordinates and a local slab bending angle, this
    function constructs a parallel/offset surface at a distance `lt` (e.g., oceanic
    crust thickness or slab thickness), representing an internal layer boundary
    within the subducting plate.
    
    Parameters
    ----------
    sx : NDArray[np.float64]
        x-coordinates of the slab top surface.
    sy : NDArray[np.float64]
        y-coordinates of the slab top surface.
    th : float
        Local slab bending angle associated with the surface points (degrees or
        radians depending on the implementation; must be consistent with the
        trigonometric functions used).
    lt : float
        Layer thickness used to offset the surface (e.g., oceanic crust thickness
        or slab thickness) (SI units: [m]).
    
    Returns
    -------
    cx : NDArray[np.float64]
        x-coordinates of the layer-defining surface.
    cy : NDArray[np.float64]
        y-coordinates of the layer-defining surface.
    """
    from stonedfenicsx.create_mesh.aux_create_mesh import _find_e_node,_correct_
    cx = np.zeros([np.amax(sx.shape)])
    cy = np.zeros([np.amax(sx.shape)])
        # ── arc-length along the top surface ─────────────────────────────────────
    ds    = np.sqrt(np.diff(sx)**2 + np.diff(sy)**2)
    dth = np.diff(th)
    dth_ds = dth/ds                  # [rad / m]


    
    kappa = np.zeros_like(sx)
    kappa[1:] = np.abs(dth_ds)
    
    if 0.8 * np.min(1/kappa) < lt: 
        lt = np.floor(0.8 * np.min(1/kappa))
    
  
    # Loop over the interface of the slab and find the points on the top of the surface of the subduction channel: the point on the of the top of the channel are perpendicular to the slab interface#
    # Compute the top surface of the subduction channel
    cx = sx - lt*np.sin(th)
    cy = sy - lt*np.cos(th)

    e_node2,t_ex1 = _find_e_node(cx,cy,cx*0.0,-np.min(sy),flag=False)
    cx,cy,t  = _correct_(cx,cy,e_node2,-np.min(sy),cy*0.0,0.0)

    cx_n = cx[(cx>=0.0) & (cy>=np.min(sy))]
    cy_n = cy[(cx>=0.0) & (cy>=np.min(sy))]
    # Shift the first node: 
    
    if cx_n[0] != 0.0: 
        cx_n[0] = 0.0 
        cy_n[0] = -lt/np.cos(th[0])
    
    
    

    return cx_n,cy_n
    
#-----------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    from stonedfenicsx.create_mesh.aux_create_mesh import find_slab_surface, Geom_input 
    from stonedfenicsx.Stoned_fenicx import fill_geometrical_input
    from stonedfenicsx.utils import parse_input
    
    path_input = "/Users/wlnw570/Work/Leeds/Fenics_tutorial/input.yaml"

    inp,Ph = parse_input(path_input)
    inp.van_keken = 0 
    g_input = fill_geometrical_input(inp)
       
    g_input.sub_type = 'Real'
    g_input.sub_path = '/Users/wlnw570/Work/Leeds/Fenics_tutorial/example_slab_surfaces/Mexico_slab.pz'
    g_input.sub_constant_flag = 0
    
    slab_top, theta_mean = find_slab_surface(g_input)
    bx,by = generate_parallel_layer_subducting_plate(slab_top[:,0],slab_top[:,1],theta_mean,130e3)
    
    
    
    
    
    
