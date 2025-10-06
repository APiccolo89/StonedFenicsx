import numpy as np

class Slab():
    def __init__(self,num_segment=[],theta_0=[],theta_max=[],D0=[],L0=[],Lb=[],y_min=[],dl = 10 ,trench=0.0,flag_constant_theta:bool=False):
        """
        Class containing the information of the subducting plate
        
        """
        self.D0 = D0
        self.L0 = L0
        self.Lb = Lb
        self.dl = dl
        self.trench = trench
        self.theta_0   = theta_0*np.pi/180
        self.theta_max = theta_max*np.pi/180
        self.num_segment = 10
        self.y_min = y_min/1e3 # convert to km to lazy to change a few stuff here
        self.slab_mid = np.zeros((self.num_segment+1,2))
        self.slab_top = np.zeros((self.num_segment+1,2))
        self.slab_bot = np.zeros((self.num_segment+1,2))
        self.theta_mean = np.zeros(self.num_segment+1)
        self.flag_constant_theta = flag_constant_theta
        self.normal_vector = np.zeros((self.num_segment+1,2)) # normal vector
        self.tangent_vector = np.zeros((self.num_segment+1,2)) # tangent vector
    def compute_bending_angle(self,l: float):
        
        """
        Compute the bending angle as a function of L
        """
        if self.flag_constant_theta:
            theta = self.theta_max
        else:
            if l > self.Lb:
                theta = self.theta_max
            else:
                theta = _ribe_angle(self.theta_max, self.Lb, l)
                if theta<self.theta_0: 
                    theta = self.theta_0
        
        return theta


    def _find_slab_surface(self):
        """
        Find the slab boundary condition.
        """
        if self.flag_constant_theta:
        # compute the bending angle as a function of L
            self.slab_mid[0,0] = self.trench     
            self.slab_mid[0,1] = 0.0
            self.slab_top[0,0] = self.trench
            self.slab_top[0,1] = 0.0
            self.slab_bot[0,0] = self.trench
            self.slab_bot[0,1] = 0.0        
        else:
            self.slab_mid[0,0] = self.trench     
            self.slab_mid[0,1] = -self.D0/2
            self.slab_top[0,0] = self.trench
            self.slab_top[0,1] = 0.0
            self.slab_bot[0,0] = self.trench
            self.slab_bot[0,1] = -self.D0        
        
        l = 0.0
        ln = 0.0
        if self.dl == []:
            dl = np.float64(self.L0/(self.num_segment))
        else:
            dl = self.dl
        
        it = 0 
    

        statement = True 
        while statement == True:
            ln += dl

            theta1 = self.compute_bending_angle(l) # bending angle at the beginning of the segment
            theta2 = self.compute_bending_angle(ln) # bending angle at the end of the segment
            theta_mean = 0.5*(theta1+theta2) # mean bending angle
            self.theta_mean[it]= theta_mean
            theta_meani_1 = theta_mean

            # Find the middle of the slab
            slab_topi_ix = self.slab_top[it,0]+dl*(np.cos(theta_mean)) # middle of the slab at the end of the segment x
            slab_topi_iz = self.slab_top[it,1]-dl*(np.sin(theta_mean)) # middle of the slab at the end of the segment z

            # Find normal vector of the segment
            #self.normal_vectori_x = -np.sin(theta_mean)
            #self.normal_vectori_z = np.cos(theta_mean)
            ## Find tangent vector of the segment
            #self.tangent_vector = np.cos(theta_mean)
            #self.tangent_vector[it,1] = -np.sin(theta_mean)
            # Find the top and bottom of the slab
         

            if it+1 > len(self.slab_mid[:,0])-1:
                self.slab_top = np.vstack([self.slab_top,[slab_topi_ix,slab_topi_iz]])
                self.theta_mean = np.hstack([self.theta_mean,theta_meani_1])
            else: 
                self.slab_top[it+1,0] = slab_topi_ix
                self.slab_top[it+1,1] = slab_topi_iz
                self.theta_mean[it] = theta_mean
                self.theta_mean[it+1] = theta_mean


        
        
            if self.y_min != -1e23: 
                x = self.slab_top[it+1,1]
                statement = x > self.y_min
                if statement == False:
                    dz = self.slab_top[it,1] - self.y_min
                    dx = dz / np.tan(theta_mean)
                    self.slab_top[it+1,0] = self.slab_top[it,0] + dx
                    self.slab_top[it+1,1] = self.y_min
            it = it+1

            l = ln

            if self.y_min == -1e23:
                statement = it < self.num_segment 


        print('New number of segments:',it-1)
        self.num_segment = it-1
        print('New length of the slab:',l)
        self.L0 = l
        
        return self


def _ribe_angle(theta_max: float, Lb: float, l: float) -> float:

    theta = theta_max*l**2*(3*Lb-2*l)/(Lb**3)

    return theta