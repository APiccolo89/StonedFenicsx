from stonedfenicsx.package_import import * 

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import numpy as np

from scipy.special import expit

from scipy.interpolate import CubicSpline


def apply_Savitzky_Golay(yd,order,cf):
    
    from scipy.signal import savgol_filter as sgf 
    
    yd_smooth = sgf(yd,cf,order)
    
    return yd_smooth

    
    
    
    
    



def moving_average_distance(y, order, window):
    """
    Distance-based moving average.

    Parameters
    ----------
    x : 1D array (must be sorted)
    y : 1D array
    window : float (physical window size)

    Returns
    -------
    y_smooth : 1D array
    """
    y_smooth = np.zeros_like(y)
    half_w = window / 2

    for i in range(len(x)):
        mask = np.abs(x - x[i]) <= half_w
        y_smooth[i] = np.mean(y[mask])

    return y_smooth

def curve_fitting(xd:NDArray[float],yd:NDArray[float])->NDArray[float]:
    """
    """
    def quadratic(x,a,b,c):
        return a * x**2 + b * x + c
    
    def cubic(x,a,b,c,d):
        return a * x**3 + b * x**2 + c * x + d
    
    def quartic(x,a,b,c,d,e):
        return a * x**4 + b * x**3 + c * x**2 + d * x + e   
    
    def quintic(x,a,b,c,d,e,f): 
        return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f  
    
    def logistic_function(x,A,L,k,a):
        return A+ (L-A)/(1+expit(k * (x - a)))
    
    def spline_function(x,y): 
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(x,y)
        
        x_range = np.linspace(np.min(x), np.max(x), 10000)
        plt.plot(x_range, cs(x_range), label='Cubic Spline')
        y_fit = cs(x_range)
        
        return y_fit
    
    def misfit(a,b):
        
        
        
        return np.sqrt(np.mean((a-b)**2))/(np.max(a)-np.min(a))
    
    def wrapper_function(f:callable,x_scale:float,y_scale:float , *args):
        def f2(x): 
            return y_scale * f(x/x_scale,*args)
        
        return f2 
    
    def perform_fitting(x:NDArray[float],y:NDArray[float],f:callable, theta_fit:bool = False):
        
        # Normalise x and y
        xs = np.abs(x.copy())/np.max(np.abs(x))
        ys = np.abs(y.copy())/np.max(np.abs(y))
                
        fit_curve, prm = curve_fit(f,xs,ys)
        
        y_ft = f(xs,*fit_curve) * np.max(np.abs(y)) * np.sign(y[-1])
        if not theta_fit:
            theta, ell, _, _  = compute_bending_angle_length(x,y_ft) 
        else:
            theta = None
            ell = None
        
        return  theta, ell, fit_curve, y_ft        

    def compute_misfit(a:NDArray[float],b:NDArray[float],c:NDArray[float],R0:NDArray[float],R1:NDArray[float],R2:NDArray[float])->tuple[float,float,float]:
        
        # compute misfit geometry
        
        r0 = misfit(R0,a)
        
        # compute misfit bending angle
        r1 = misfit(R1,b)
        
        # compute misfit total length 
        
        r2 = misfit(R2,c)
        
    
        
        
        return r0, r1, r2
        
        
    def function_bending_data(ell:NDArray[float],theta:NDArray[float])->float:
        
        cs=CubicSpline(ell,theta)
        
        
        def f2(x:float| NDArray[float]):
            
            return cs(x)
        
    
        return f2  
    
    
    
    theta, ell, xd,yd = compute_bending_angle_length(xd,yd)

    theta_sgf = apply_Savitzky_Golay(theta,2,50)        
    
    f_theta = function_bending_data(ell,theta_sgf)

    slab_top,theta_mean,ell_s = create_slab_surface(f_theta,np.min(yd),stp = 10.0)
    
    
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(ell,theta,c='k',alpha=0.1,label='original')
    ax.plot(ell,theta_sgf, c='b', label = 'filtered data')
    ax.plot(ell_s,theta_mean,c='r',label='post processed')
    plt.legend()
    
    
    
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(xd,yd,c='k',alpha=0.1,label='original')
    ax.plot(slab_top[:,0],slab_top[:,1],c='r',label='post processed')
    plt.legend()
    
    
    

    



    return np.array([0.0,0.0])




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
    alpha = (theta[0]-1.0)/dl

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
    
    slab= np.loadtxt(file_path)
    ax = slab[:,0]
    ay = -slab[:,1]
    dz = slab[:,2]
    
    ax = ax[ay<-10]
    dz = dz[ay<-10]
    ay = ay[ay<-10]

    a = curve_fitting(ax,ay)


    return ax, ay 



def create_slab_surface(f:callable, y_min:float,stp=float)->tuple[NDArray[float],NDArray[float]]:
    
    # Initialise the theta_mean and slab_top array
    theta_mean = np.zeros([2],dtype=float)
    ell_s = np.zeros([2],dtype=float)
    slab_top   = np.zeros([2,2],dtype=float)
        
    # compute the bending angle as a function of L
    slab_top[0,0] = 0.0
    slab_top[0,1] = 0.0
    dl = stp
    lghn = 0.0 
    lgh = 0.0
    it = 0 
    statement = True 
    while statement:
        lghn += dl
        theta1 = f(lgh) # bending angle at the beginning of the segment
        theta2 = f(lghn) # bending angle at the end of the segment
        theta = theta1 # mean bending angle
        theta_mean[it]= theta
        theta_meani_1 = theta
        # Find the middle of the slab
        slab_topi_ix = slab_top[it,0]+dl*(np.cos(theta)) # middle of the slab at the end of the segment x
        slab_topi_iz = slab_top[it,1]-dl*(np.sin(theta)) # middle of the slab at the end of the segment z
        ell_s[it] = lgh

        if it+1 > len(slab_top[:,0])-1:
            slab_top = np.vstack([slab_top,[slab_topi_ix,slab_topi_iz]])
            theta_mean = np.append(theta_mean,theta_meani_1)
            ell_s = np.append(ell_s,lgh)
        else: 
            slab_top[it+1,0] = slab_topi_ix
            slab_top[it+1,1] = slab_topi_iz
            theta_mean[it] = theta
            theta_mean[it+1] = theta
        if y_min != -1e23: 
            x = slab_top[it+1,1]
            statement = x > y_min
            if not statement:
                dz = slab_top[it,1] - y_min
                dx = dz / np.tan(theta)
                slab_top[it+1,0] = slab_top[it,0] + dx
                slab_top[it+1,1] = y_min
        it = it+1
        lgh = lghn

    
    
    
    return slab_top,theta_mean,ell_s



if __name__ == '__main__': 
    
    read_file_slab('example_slab_surfaces/Mexico_slab.pz')
    
    
