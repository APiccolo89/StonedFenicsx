from stonedfenicsx.package_import import np,NDArray
from .aux_create_mesh import Geom_input



def find_slab_surface(g_input:Geom_input)->tuple([NDArray[float],NDArray[float]]):  
    """
    Compute the top surface of a kinematic slab as a polyline starting from the trench.

    The slab surface is discretised into straight segments of length `g_input.sub_dl`.
    At each step we compute the local bending angle at the current and next arc-length
    positions, average them, and use that mean angle to advance to the next point.

    Angle convention
    ----------------
    theta is measured with respect to the positive horizontal x-axis.

             theta
    x-axis  -------\------
                    \ theta
                     \/

    Algorithm (summary)
    -------------------
    1. Initialise `top_slab` with the trench point.
    2. Initialise the arc-length `lgh = 0.0` (distance measured along the slab surface).
    3. While the current point is above the model bottom boundary (`y > ymin`):
       a. Set `lghn = lgh + g_input.sub_dl`.
       b. Compute `theta  = theta(lgh)`  and `theta1 = theta(lghn)`.
       c. Compute `theta_mean = 0.5 * (theta + theta1)`.
       d. Use `theta_mean` to advance one step and append the new point to `top_slab`.
       e. Update `lgh = lghn`.

    Returns
    -------
    top_slab : (n_segment, 2) ndarray
        Coordinates (x, y) of the slab top surface polyline.
    theta_mean : float
        Mean bending angle used for the last segment (or an average over segments,
        depending on your implementation).

    Raises
    ------
    ValueError
        If the selected slab-surface method is not implemented (only "custom" is
        currently supported).
    """

 
    if g_input.sub_type == 'Costum':
        # Initialise the theta_mean and slab_top array
        theta_mean = np.zeros([2],dtype=float)
        slab_top   = np.zeros([2,2],dtype=float)
        
        # compute the bending angle as a function of L
        slab_top[0,0] = g_input.sub_trench[0]
        slab_top[0,1] = 0.0

        lgh = 0.0
        lghn = 0.0

        dl = g_input.sub_dl

        it = 0 

        statement = True 
        while statement:
            lghn += dl
            theta1 = compute_bending_angle(g_input,lgh) # bending angle at the beginning of the segment
            theta2 = compute_bending_angle(g_input,lghn) # bending angle at the end of the segment
            theta = 0.5*(theta1+theta2) # mean bending angle
            theta_mean[it]= theta
            theta_meani_1 = theta
            # Find the middle of the slab
            slab_topi_ix = slab_top[it,0]+dl*(np.cos(theta)) # middle of the slab at the end of the segment x
            slab_topi_iz = slab_top[it,1]-dl*(np.sin(theta)) # middle of the slab at the end of the segment z


            if it+1 > len(slab_top[:,0])-1:
                slab_top = np.vstack([slab_top,[slab_topi_ix,slab_topi_iz]])
                theta_mean = np.append(theta_mean,theta_meani_1)
            else: 
                slab_top[it+1,0] = slab_topi_ix
                slab_top[it+1,1] = slab_topi_iz
                theta_mean[it] = theta
                theta_mean[it+1] = theta
            if g_input.y[0] != -1e23: 
                x = slab_top[it+1,1]
                statement = x > g_input.y[0]
                if not statement:
                    dz = slab_top[it,1] - g_input.y[0]
                    dx = dz / np.tan(theta)
                    slab_top[it+1,0] = slab_top[it,0] + dx
                    slab_top[it+1,1] = g_input.y[0]
            it = it+1
            lgh = lghn
            
    else: 
        raise ValueError("There is not yet any alternative to costum subduction yet, please, be patient")
        
    return slab_top, theta_mean


def compute_bending_angle(g_input:Geom_input
                        ,lgh: float):
    
    """compute_bending_angle: 
    inputs: 
    geometrical information
    current length of the slab 
    Returns:
        theta:float -> for a given bending angle function, return theta = f(l) -> l the actual distance along the slab surface from the trench
    """
    if g_input.sub_constant_flag:
        theta = g_input.sub_theta_max
    else:
        if lgh > g_input.Lb:
            theta = g_input.sub_theta_max
        else:
            theta = ribe_angle(g_input.sub_theta_max, g_input.sub_LbLb, lgh)
            if theta<g_input.sub_theta_0: 
                theta = g_input.sub_theta_0
    
    return theta


def ribe_angle(theta_max: float
                ,Lb: float
                ,lgh: float) -> float:

    """ribe_angle 
    inputs: 
    theta_max : float -> maximum angle of the slab
    Lb : float -> critical along slab distance where the bending is occuring 
    lgh : float -> current position along the slab surface  

    Returns:
        theta :float -> Bending angle at the local point along the top surface slab. 
    """

    theta = theta_max*lgh**2*(3*Lb-2*lgh)/(Lb**3)

    return theta