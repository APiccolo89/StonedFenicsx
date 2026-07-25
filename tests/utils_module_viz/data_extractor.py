import h5py
import numpy as np
import os
r_grid = 800
def get_nested_attr(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj

class Test:
    def __init__(self,path_2_test:str):
        self.meta_data = None
        self.path_2_test = path_2_test 
        self.MeshData = MeshData(self.path_2_test)
        self.Data_raw = Data_Raw(self.path_2_test,num=len(self.MeshData.X[:,0]))
        
    def interpolate_data(self,Data_field):
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
        values =get_nested_attr(self.Data_raw, Data_field)
        
        Zi = griddata(self.MeshData.X, values, (Xi, Yi), method='linear')
        
        Zi[self.MeshData.ar==False] = np.nan

        
        return Zi  
    
    def _interpolate_data_slab(self,Data_field):
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
        values =get_nested_attr(self.Data_raw, Data_field)
        
        Zi = griddata(self.MeshData.X, values, (Xi, Yi), method='linear')
        
        Zi[not self.MeshData.ar_s] = np.nan

        
        return Zi  

    def _interpolate_data_ex(self,field):
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
        Zi = griddata(self.MeshData.X, field, (Xi, Yi), method='linear')
        
        Zi[not self.MeshData.ar] = np.nan

        
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
        
#        self.TimeDependent = Data_experiment(f,num,ts=True)
        
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
        
        if ts: 
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
            times = list(fl[field].keys())
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
        self.T_Ad    = np.zeros([num,TS],dtype=float) 
            
        
        self.extract_data(f,ts)
        
        
    def extract_data(self,f:str,ts:bool):
        import h5py
        import numpy as np
        
        if ts: 
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
            f = '%s/Steady_State.h5'%f

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
            
            self.k  = np.array(fl['/Function/k  [W/m/K]/0']).flatten()
            
            self.rho = np.array(fl['/Function/Density  [kg/m3]/0']).flatten()
            
            self.eta = np.array(fl['/Function/Viscosity  [Pa.s]/0']).flatten()
            
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
        f = '%s/Steady_State.h5'%f
        
        # Extract mesh geometry 
        fl = h5py.File(f, 'r')

        X                        = np.array(fl['/Mesh/mesh/geometry'])
    
        ar_point = np.array(fl['Function/MeshTAG/0']) 
        
        ind = np.where(ar_point!=0.0)
        
        ind = ind[0]
        
        self.mesh_tag = ar_point.flatten()  
        
        self.X                   = X
        
        self.xi                  = np.linspace(np.min(X[:,0]),np.max(X[:,0]),r_grid)
        
        self.yi                  = np.linspace(np.min(X[:,1]),np.max(X[:,1]),r_grid)
        
        self.Xi,      self.Yi     = np.meshgrid(self.xi,self.yi)
        
        self.polygon, self.ar     = self.create_polygon()
        
        self.polygon_S,self.ar_s = self.create_polygon_slab()
        
        self.ind_topSlab  = (self.mesh_tag==8.0)| (self.mesh_tag==9.0)
    
        self.ind_Oceanic  = (self.mesh_tag==10.0)

        fl.close()

    
    def create_polygon_slab(self): 

        def extract_coordinates(mesh_tag:list)->np.ndarray:
            
            if len(mesh_tag)==2:
                condition = (self.mesh_tag == mesh_tag[0]) | (self.mesh_tag == mesh_tag[1])
            else: 
                condition = self.mesh_tag == mesh_tag[0]
            
            xbuf = self.X[condition,0]
            
            ybuf = self.X[condition,1]
            
            sort = np.argsort(xbuf)
            
            X = np.array([xbuf[sort], ybuf[sort]])
            
            X = np.transpose(X)
            
            return X 
        

        
        x = self.X[:,0]
        
        y = self.X[:,1]

        # Left inlet: 
        X_L = extract_coordinates([7])
        # Bottom slab:
        X_BS = extract_coordinates([6])
        #
        X_BT = extract_coordinates([5])
        # Slab
        X_Sl = extract_coordinates([9,8])
        
        a = X_Sl[:,0]
        b = X_Sl[:,1]
        a = a[::-1]
        b = b[::-1]
        X_Sl[:,0] = a 
        X_Sl[:,1] = b
    
        
        from shapely.geometry import Polygon as sPolygon
        
        from shapely import contains_xy as scontains_xy
        
        polygon = sPolygon(np.vstack((np.array(X_L), np.array(X_BS), np.array(X_BT), np.array(X_Sl))))
        
        ar =  scontains_xy(polygon,self.Xi,self.Yi)        
        
        
       
        return polygon, ar
    
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

        
        xbt = self.X[(self.mesh_tag==6.0),0]
        ybt = self.X[(self.mesh_tag==6.0),1]
        sort = np.argsort(xbt)
    
        p_list  = np.array([xbt[sort], ybt[sort]])
        
        p_list = p_list.transpose()
            
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
        
        top = top[ind_arg[::-1], :]        
        
        from shapely.geometry import Polygon as sPolygon
        
        from shapely import contains_xy as scontains_xy
        
        polygon = sPolygon(np.vstack((np.array(p_list), np.array(bottom), np.array(right), np.array(top), np.array(left))))
        
        ar =  scontains_xy(polygon,self.Xi,self.Yi)

        return polygon, ar 