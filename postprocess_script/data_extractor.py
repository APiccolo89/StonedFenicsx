import h5py
import numpy as np
import os
class Test:
    def __init__(self,path_2_test:str):
        self.meta_data = None
        self.path_2_test = path_2_test 
        self.MeshData = MeshData(self.path_2_test)
        self.Data_raw = Data_Raw(self.path_2_test,num=len(self.MeshData.X[:,0]))
        
    def _interpolate_data(self,Data_field):
        """ Interpolate the data on a regular grid for visualisation
        Args:
            Data_field (str): field to be interpolated
        Returns:
            Zi (np.array): interpolated data
            
            Paraview is not able to visualise unstructured data properly, so we need to interpolate the data on a regular grid for visualisation purposes.
            The nan mask is used to mask the data outside the domain, and giving you the illusion that I was able to generate a curved mesh in python. Ahah. 
        """
        import numpy as np
        from scipy.interpolate import griddata
        
        xi = self.MeshData.X[:,0]
        yi = self.MeshData.X[:,1]
        Xi = self.MeshData.Xi
        Yi = self.MeshData.Yi
        
        # Extrac the field data 
        values = eval('self.Data_raw.%s'%Data_field)
        
        Zi = griddata(self.MeshData.X, values, (Xi, Yi), method='linear')
        
        Zi[self.MeshData.ar==False] = np.nan

        
        return Zi  


class Data_Raw():
    def __init__(self,f:str,num:int):
        """ Extract the data from the h5 file. 

        Args:
            f (str): path to the test 
        """
        
        import h5py
        import numpy as np
        
        self.SteadyState = Data_experiment(f,num,ts=False)
        
        self.TimeDependent = Data_experiment(f,num,ts=True)
        
class Data_experiment():
    '''
    Class to extract the data from either the steady state or time dependent h5 file.
    the init function requires the path to the test, the number of points and the number of timestep
    The main issue is for the timedependent case, as I need to extract the number of timesteps from the 
    the h5 file, but I will create the function later, and will be in the metadata field of the test 
    '''
    def __init__(self,f:str,num:int,ts:bool):
        """ Extract the steady state data from the h5 file. 

        Args:
            f (str)  : path to the test
            num (int): number of points 
            ts (int) : time step
        """
        
        import h5py
        import numpy as np
        
        self.times     = None
        self.time_list = None
        self.Temp      = None
        self.Pres      = None
        self.LitPres   = None
        self.vx        = None
        self.vy        = None
        self.qx        = None
        self.qy        = None  
        self.kappa     = None 
        self.alpha     = None 
        self.eta       = None 
        self.rho       = None 
        self.k         = None 
        self.Cp        = None
        self.NoTD      = False
        
        if ts == True: 
            # Rather necessary as the h5 file was not created in reasonable way 
            # Direct to the time dependent file
            if os.path.exists('%s/TimeDependent.h5'%f):
                print("The file exists.")
            else:
                print("The file does not exist. Either the time dependent simulation was not run or the path is incorrect.")
                self.NoTD = True
                return None 
            
            fl = h5py.File('%s/TimeDependent.h5'%f, 'r')
            field = 'Function/Temperature  [degC]'
            times = list(f[field].keys())
            time_list = [float(s.replace("_", ".")) for s in times]
            time_sort = np.argsort(time_list)
            time_list = [time_list[i] for i in time_sort]
            times = [times[i] for i in time_sort]
            TS = len(times) 
            self.times = times
            self.time_list = time_list
            fl.close()
        else: 
            TS = 0 
        
        self.Temp    = np.zeros([num,TS],dtype=float)
        self.Pres    = np.zeros([num,TS],dtype=float)
        self.LitPres = np.zeros([num,TS],dtype=float)
        self.vx      = np.zeros([num,TS],dtype=float)
        self.vy      = np.zeros([num,TS],dtype=float)
        self.qx      = np.zeros([num,TS],dtype=float)
        self.qy      = np.zeros([num,TS],dtype=float) 
        self.kappa   = np.zeros([num,TS],dtype=float) 
        self.alpha   = np.zeros([num,TS],dtype=float) 
        self.eta     = np.zeros([num,TS],dtype=float) 
        self.rho     = np.zeros([num,TS],dtype=float) 
        self.k       = np.zeros([num,TS],dtype=float) 
        self.Cp      = np.zeros([num,TS],dtype=float) 
            
        
        self.extract_data(f,ts)
        
        
    def extract_data(self,f:str,ts:bool):
        import h5py
        import numpy as np
        
        if ts == True: 
            # Direct to the time dependent file
            fl = h5py.File('%s/TimeDependent.h5'%f, 'r')
            for it, time in enumerate(self.times):
                field_temp = '/Function/Temperature  [degC]/%s'%time
                field_pres = '/Function/Pressure  [GPa]/%s'%time
                field_litpres = '/Function/Lit Pres  [GPa]/%s'%time
                field_v   = '/Function/Velocity  [cm/yr]/%s'%time
                field_q   = '/Function/q  [W/m2]/%s'%time
                
                self.Temp[:,it]    = np.array(fl[field_temp]).flatten()
                self.Pres[:,it]    = np.array(fl[field_pres]).flatten()
                self.LitPres[:,it] = np.array(fl[field_litpres]).flatten()
                
                v                 = np.array(fl[field_v])
                self.vx[:,it]     = v[:,0]
                self.vy[:,it]     = v[:,1]
                
                qS               = np.array(fl[field_q])
                self.qx[:,it]    = qS[:,0]
                self.qy[:,it]    = qS[:,1]
                
                fl.close()
        else:
            # Direct to the steady state file 
            f = '%s/Steady_state.h5'%f

            # Extract mesh geometry 
            fl = h5py.File(f, 'r')


            self.Temp               = np.array(fl['/Function/Temperature  [degC]/0']).flatten()

            self.Pres               = np.array(fl['/Function/Pressure  [GPa]/0']).flatten()

            self.LitPres            = np.array(fl['/Function/Lit Pres  [GPa]/0']).flatten()

            v                       =  np.array(fl['Function/Velocity  [cm/yr]/0'])

            self.vx                 = v[:,0]

            self.vy                 = v[:,1]

            qS = np.array(fl['Function/Heat flux [W/m2]/0'])

            self.qx = qS[:,0]

            self.qy = qS[:,1]
            
            self.Cp = np.array(fl['/Function/Cp  [J/kg]/0']).flatten()
            
            self.k  = np.array(fl['/Function/k  [W/m/k]/0']).flatten()
            
            self.rho = np.array(fl['/Function/Density  [kg/m3]/0']).flatten()
            
            self.eta = np.array(fl['/Function/eta  [Pa s]/0']).flatten()
            
            self.alpha = np.array(fl['/Function/alpha  [1/K]/0']).flatten()
            
            self.kappa = np.array(fl['/Function/kappa  [m2/s]/0']).flatten()

            fl.close()
        
class MeshData(): 
    def __init__(self,f:str):
        """ Extract the mesh data from the h5 file. Produce the visualisation grid and the polygon 

        Args:
            f (str): path to the test 
        """
        

        
        # Direct to the steady state file 
        f = '%s/Steady_state.h5'%f
        
        # Extract mesh geometry 
        fl = h5py.File(f, 'r')

        X                        = np.array(fl['/Mesh/mesh/geometry'])
        
        self.X                   = X
        
        self.xi                  = np.linspace(np.min(X[:,0]),np.max(X[:,0]),2000)
        
        self.yi                  = np.linspace(np.min(X[:,1]),np.max(X[:,1]),2000)
        
        self.Xi,      self.Yi     = np.meshgrid(self.xi,self.yi)
        
        self.polygon, self.ar     = self.create_polygon()
        
        ar_point = np.array(fl['Function/MeshTAG/0']) 
        
        ind = np.where(ar_point!=0.0)
        
        ind = ind[0]
        
        self.mesh_tag = ar_point.flatten()
        
        self.ind_topSlab  = (self.mesh_tag==8.0)| (self.mesh_tag==9.0)
    
        self.ind_Oceanic  = (self.mesh_tag==10.0)

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