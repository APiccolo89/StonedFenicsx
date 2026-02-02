import numpy as np 
from Subducting_plate import Slab 


S = Slab(100,0.0,70,100,800,200,-660,dl=10,trench=0.0,flag_constant_theta=False,depth_flattening=[-400,-560])
S._find_slab_surface()