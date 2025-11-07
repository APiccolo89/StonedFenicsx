import vtk 
from mpi4py import MPI
import numpy as np
import dolfinx 
import h5py
from dolfinx.io import XDMFFile
import os
import cmcrameri

class MeshData(): 
    def __init__(self,f:h5py.File):
        # Extract mesh geometry 
        fl = h5py.File(td_file, 'r')

        X                        = np.array(fl['/Mesh/mesh/geometry'])
        
        self.X                   = X
        
        self.xi                  = np.linspace(np.min(X[:,0]),np.max(X[:,0]),2000)
        
        self.yi                  = np.linspace(np.min(X[:,1]),np.max(X[:,1]),2000)
        
        self.Xi,      self.Yi     = np.meshgrid(self.xi,self.yi)
        
        self.polygon, self.ar     = self.create_polygon()
        
        fl.close()
       
        
    def create_polygon(self):
        
        x_min = np.min(self.X[:,0])
        
        x_max = np.max(self.X[:,0])
        
        y_min = np.min(self.X[:,1])
        
        y_max = np.max(self.X[:,1])
        
        x = self.X[:,0]
        
        y = self.X[:,1]

        top    = np.array([x[self.X[:,1]==y_max],y[ self.X[:,1]==y_max]])
        
        bottom = np.array([x[self.X[:,1]==y_min],y[ self.X[:,1]==y_min]])
        
        left   = np.array([x[self.X[:,0]==x_min],y[ self.X[:,0]==x_min]])
        
        right  = np.array([x[self.X[:,0]==x_max],y[ self.X[:,0]==x_max]])
        
        x_bot  = np.min(bottom[0,:])
        
        l_min  = np.min(left[1,:])
        
        p0     = np.array([x_min, l_min])
        
        p_list = []
        
        p_list.append(p0)
        
        it     = 0
        
        p      = p0 
         
        while p[0] != x_bot and p[1] != y_min:
            """
            Ora, sono molto stupido, e lo riconosco, quindi ho costruito questo aborto usando un po' trial and error. Parto dal punto piu' basso a sinistra,
            da li vado a cercare i punti che hanno una distanza minore di 6 km, tra questi scelgo quelli che hanno il minore y del punto precedente, e poi seleziono
            le nuove coordinate del punto che verra' usato per la successiva iterazione. Correzione dovuta, arg sort in funzione di y. L'idea e' stata modificata 
            a tentativi, perche', ripeto, non sono una persona brillante e probabilmente avrei dovuto essere abortito prima di nascere. 
            """


            d = np.sqrt((x-p_list[it][0])**2 + (y-p_list[it][1])**2)
          
            idx = np.where((d < 6.0))   
          
            x_pr = x[idx[0]];y_pr = y[idx[0]];d_pr = d[idx[0]]
          
            x_pr = x_pr[y_pr<p_list[it][1]];d_pr = d_pr[y_pr<p_list[it][1]];y_pr = y_pr[y_pr<p_list[it][1]]
          
            d_pr_sort = np.argsort(y_pr)
          
            d_pr = d_pr[d_pr_sort];x_pr = x_pr[d_pr_sort];y_pr = y_pr[d_pr_sort]
          
            x_new = x_pr[0]
          
            y_new = y_pr[0]
          
            p = np.array([x_new,y_new])
          
            p_list.append(p)
                    
            it = it + 1 
            
        bottom = bottom.transpose()
        
        right = right.transpose()
        
        top = top.transpose()
        
        left = left.transpose()
        # order bottom boundary 
        
        x_bottom = bottom[:,0]
        
        ind_arg = np.argsort(x_bottom)
        
        bottom = bottom[ind_arg,:]
        # order right boundary
        
        y_right = right[:,1]
        
        ind_arg = np.argsort(y_right)
        
        right = right[ind_arg,:]
        # order top boundary
        
        x_top = top[:,0]
        
        ind_arg = np.argsort(x_top)
        
        top = top[np.invert(ind_arg),:]
        
        from shapely.geometry import Polygon as sPolygon
        
        from shapely import contains_xy as scontains_xy
        
        polygon = sPolygon(np.vstack((np.array(p_list), np.array(bottom), np.array(right), np.array(top), np.array(left))))
        
        ar =  scontains_xy(polygon,self.Xi,self.Yi)

        return polygon, ar 
        
            
        
        
        
        

def create_figure(path2save:str, 
                  time_string, 
                  vmin:float, 
                  vmax:float, 
                  cmap: str, 
                  title: str,
                  Xi:float, 
                  Yi:float, 
                  field:float, 
                  n_level:int,
                  name_fig:str,
                  ipic:int): 
    
    import matplotlib.pyplot as plt

    plt.rcParams.update({
    "text.usetex": True,           # Use LaTeX for all text
    "font.family": "serif",        # Or 'sans-serif'
    "font.serif": ["Computer Modern"],   # LaTeX default
    "axes.unicode_minus": False,
    })
    def modify_ax(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(direction='in', which='both', length=6, width=1, colors='black', grid_color='black', grid_alpha=0.5)
        ax.tick_params(axis='x', direction='in', which='both', bottom=True, top=False)
        ax.tick_params(axis='y', direction='in', which='both', left=True, right=False)
        return ax


    pt_save = os.path.join(path2save,name_fig)
    if not os.path.isdir(pt_save):
        os.makedirs(pt_save)
        
    figure_name = f'Figure_{ipic:03d}.png'
        
    fname = os.path.join(pt_save, )
    fig, ax0 = plt.subplots(figsize=(10, 6))
    ax0 = modify_ax(ax0)
    ax0.set_title(time_string, fontsize=16)
    ax0.set_xlabel('Distance [km]', fontsize=14)
    ax0.set_ylabel('Depth [km]', fontsize=14)
    p0 = ax0.contourf(Xi, Yi, field, levels=n_level, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(p0, ax=ax0, orientation='vertical', pad=0.02)
    cbar.set_label(label=title, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks(np.linspace(vmin, vmax, 5))
    cbar.set_ticklabels([f'{val:.0f}' for val in np.linspace(vmin, vmax, 5)])
    fig.savefig(fname)
    
    return 0 

def compare_SS_TD(ss_file:str, td_file:str, time_td, M_data:MeshData,path_2_save:str):

    
    """
    Easy peasy function to compare steady state and time dependent solution at given time. Moreover, to make nice plot rather than 
    that crap from paraview. 
    1st: load the timedependent file and extract the geometry and temperature at given time
         1. find the outerboundaries of the domain (very ugly way)
         2. create a polygon from these boundaries
         3. create a grid and interpolate the temperature on it
         4. mask the temperature outside the polygon
         
    2nd: load the steady state file and extract the temperature
         1. find the outerboundaries of the domain (very ugly way)
         2. create a polygon from these boundaries
         3. create a grid and interpolate the temperature on it
         4. mask the temperature outside the polygon
    3rd: make the plot
    4th: save the plot
    """
    
    
    f = h5py.File(td_file, 'r')
    field = 'Function/Temperature  [degC]'
    times = list(f[field].keys())
    time_list = [float(s.replace("_", ".")) for s in times]
    time_sort = np.argsort(time_list)
    time_list = [time_list[i] for i in time_sort]
    times = [times[i] for i in time_sort]

    from scipy.interpolate import griddata
    # Load steady state file    
    ipic = 0 
    for i in times:
        t = time_list[ipic]
        time_str = f'Time = {t:.3f} Myr'
        T = np.array(f[field+'/'+i])
        T_i = griddata(M_data.X, T, (M_data.Xi, M_data.Yi), method='linear', fill_value=np.nan)
        T_i = T_i[:,:,0]
        T_i[M_data.ar==False] = np.nan
        create_figure(path_2_save,time_str,20,1300,'cmc.lipari',r'Temperature [$^{\circ}C$]', M_data.Xi, M_data.Yi,T_i, 20, 'T',ipic)
        ipic = ipic + 1 
        
    
    
    
    
    
    
    return 0 


if __name__ == "__main__":
    path_2_test = '/Users/wlnw570/Work/Leeds/Output/Stonedphoenix/curved/case_Hobson_time_dependent_experiment'
    path_2_save = '/Users/wlnw570/Work/Leeds/Output/Stonedphoenix/curved/case_Hobson_time_dependent_experiment2/pic'
    if not os.path.isdir(path_2_save):
        os.makedirs(path_2_save)
    
    td_file = '%s/timeseries_all.h5'%(path_2_test)
    ss_file = '%s/time_dependent_output.vtu'
    time_td = 10.0  # Time in Myr to compare
    M_data = MeshData(td_file)
    
    
    compare_SS_TD(ss_file, td_file, time_td, M_data, path_2_save)
    
