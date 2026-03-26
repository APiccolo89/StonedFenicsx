import numpy as np 
import h5py as hpy 

@DataClass 
class Composition: 
    SiO2:float 
    MgO2:float 
    CaO2:float 
    Na2O:float 
    K2O: float 
    H2O: float 
    FeO:float 

        

class Point:
    def __init__(self,x0,y0):
        self.x0   = x0 
        self.y0   = y0 
        self.comp = 'Basalt'
        self.x : float = None 
        self.y : float = None 
        self.composition : Composition = None 
        self.P : float = None 
        self.T : float = None
        self.v : float = None
        self.water_loss :float = None 
        self.rho :float = None 
        self.drho:float = None 
    
    def interpolatePT(self, X,Y,Pmod,Tmod):
        pass 
    
    def interpolateVel(self,vx,vy): 
        pass 
    
    def update_position(self,dt):
        # -> to update to Runge Kutta 
        self.x = self.x + self.vx * dt 
        self.y = self.y + self.vy * dt
    
    def update_composition(self,ctrls):
        # Run Perplex 
        # Check if the water is higher than    
        # If so -> remove -> recompute the composition
        # -> compute drho with the previous state 
        # -> run again perplex 
        # -> update rho 
        
        pass

         
# -> store P-T-C -> create database -> useful for clustering analysis
# ----------
# Define initial position of 14 points
# Advect
# Check and correct
# Interpolate the P-T 
# -> Compute phase assemblage -> water free 
# -> Compute delta rho -> delta V 
# -> if water mode > threshold -> remove water -> recompute composition -> 
# -> compute new step -> with new phase diagram 
# -> vertical sum of water mode -> spot water release
#-------

