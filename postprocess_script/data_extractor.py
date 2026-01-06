class Test:
    def __init__(self):
        self.meta_data = None
        self.Data_raw = None 
        self.path_2_test = None 
        self.MeshData = MeshData()
        
        
class MeshData(): 
    def __init__(self,f:str):
        """ Extract the mesh data from the h5 file. Produce the visualisation grid and the polygon 

        Args:
            f (str): path to the test 
        """
        
        import h5py
        import numpy as np
        
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