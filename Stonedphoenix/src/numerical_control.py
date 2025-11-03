from dataclasses import dataclass,field 
import numpy as np
import os
from numba.experimental import jitclass
from numba import int64, float64,int32, types
from typing import Tuple, List
from typing import Optional
from numba import njit, prange

dict_temp_prob = {'Transient':0,'Steady_state':1}
dict_k = {'base': 0,
           'T_hof_Mckenzie2005': 1,
           'T_Xu_2004':2,
           'Tosi2013':3}

dict_Cp = {'base':0,
            'Bermann1988_fosterite':1,
            'Bermann1988_fayalite':2,
            'Bermann1988_fo89_fo11':3,
            'Bernann1996_fosterite':4,
            'Bermann1996_fayalite':5,
            'Bermann1996_fo89_fa11':6
            }
dict_rho = {'base':0,
            'temp_dep':1,
            'pres_dep':2}

dict_rheology = {'default':0,
                 'diffusion_only':1,
                 'composite':2,
                 'dislocation_only':3,
                 'composite_iter':4}

# Numba JIT class spec definition
spec = [('it_max', int64),
    ('tol', float64),
    ('tol_innerPic', float64),
    ('tol_innerNew', float64),
    ('relax', float64),
    ('Tmax', float64),
    ('Ttop', float64),
    ('g', float64),
    ('slab_age', float64),
    ('v_s', float64[:]),
    ('time_max', float64),  # Maximum time in seconds
    ('time_dependent_v',int64),
    ('time_max', float64),
    ('steady_state',int64),# Assuming this is a NumPy array
    ('slab_bc',int64),# Assuming this is a NumPy array
    ('decoupling',int64),
    ('van_keken',int64),
    ('van_keken_case',int64),
    ('model_shear',int64),# 1 decoupled, 0 coupled
    ('phase_wz',int64),
    ('wz_tk',float64),
    ('time_dependent',int64),
    ('dt',float64)
]

@jitclass(spec)
class NumericalControls:
    def __init__(self,
                 it_max=20,
                 tol=1e-4,
                 Ttop=0.0,
                 Tmax=1300.0,
                 g=9.81, 
                 time_max = 30, 
                 pressure_scaling=1e22/1000e3,
                 slab_age=0.0,
                 v_s = np.array([5.0,0.0], dtype=np.float64),
                 steady_state = 1,
                 relax = 0.6,
                 time_dependent_v = 0,
                 slab_bc = 1, # BC: 0 -> pipe like , 1 moving wall slab
                 decoupling = 1,
                 tol_innerPic = 1e-4,
                 tol_innerNew = 1e-7,
                 van_keken = 0,
                 van_keken_case = 0,
                 model_shear = 3, # 0 -> inactive/ 0 / linear 1/ tanh model 2
                 phase_wz = 7,
                 wz_tk = 1e3,
                 time_dependent = 0,
                 dt  = 500):  # 0 -> inactive / linear 

        # Direct initialization of class attributes
        self.it_max            = it_max 
        self.tol               = tol 
        self.relax             = relax
        self.Tmax              = Tmax + 273.15
        self.Ttop              = Ttop + 273.15
        self.g                 = g 
        self.v_s               = v_s  # Convert cm/yr to m/s
        self.slab_age          = slab_age
        self.time_max          = time_max
        self.time_dependent_v  = time_dependent_v
        self.steady_state      = steady_state
        self.slab_bc           = 1 # 1 moving wall, 0 pipe-like slab 
        self.decoupling        = decoupling # 1 decoupled, 0 coupled
        self.tol_innerPic      = tol_innerPic
        self.tol_innerNew      = tol_innerNew
        self.van_keken         = van_keken # 
        self.van_keken_case    = van_keken_case # 
        self.model_shear       = model_shear# 1 linear decoupling, 0 nonlinear decoupling
        self.phase_wz          = phase_wz
        self.wz_tk             = wz_tk
        self.time_dependent    = time_dependent
        self.dt                = dt # in years
        
    

class IOControls():
    def __init__(self, test_name:str = '', path_save:str = '', sname:str ='',ts_out:int = 10, dt_out:float = 1e6):
        self.test_name = test_name
        self.path_save = path_save
        self.sname = sname
        self.path_test = os.path.join(self.path_save,self.test_name)
        self.ts_out = ts_out
        self.dt_out = dt_out

    def generate_io(self):
        """
        Create directories if they don't exist.
        """
        if not os.path.isdir(self.path_save):
            os.makedirs(self.path_save)
        if not os.path.isdir(os.path.join(self.path_save, self.test_name)):
            os.makedirs(os.path.join(self.path_save, self.test_name))
        
        print('Directory created:', os.path.join(self.path_save, self.test_name))



spec_LHS = [
    ('dz', float64),
    ('nz', int32),
    ('alpha_g', float64),
    ('end_time', float64),
    ('depth_melt', float64),
    ('option_1D_solve', int32),
    ('dt', float64),
    ('recalculate', int32),
    ('van_keken', int32),
    ('z', float64[:]),
    ('LHS', float64[:]),
    ('LHS_var', float64[:, :]),
    ('c_age_plate', float64),
    ('c_age_var', float64[:]),
    ('flag', int32[:]),
    ('d_RHS', float64),
    ('t_res_vec',float64[:])
]

@jitclass(spec_LHS)
class ctrl_LHS:
    """
    This class stores and initializes parameters for the 1D thermal LHS problem.
    """

    def __init__(
        self,
        dz=1e3,               # spatial step
        nz=200,               # number of vertical cells
        alpha_g=3e-5,         # thermal expansivity
        end_time=80,       # end time [yr]
        depth_melt=0.0,       # depth of melt boundary
        option_1D_solve=1,    # flag to enable 1D solve
        dt=5e-3,              # time step
        c_age_plate=50.0,     # characteristic plate age
        c_age_var=(30.0, 60.0),  # variation in plate age
        t_res=1000,           # temporal resolution
        recalculate=0,        # flag for recomputation
        van_keken=1,          # benchmark flag
        d_RHS=-50e3,
        z_min= 140e3# distance for RHS term
    ):
        if dt > 0.1: 
            raise ValueError('dt: The input data must be in Myr. This timestep will be blow up the system. As a general remark: all input SI is Myr for time related parameters')
        elif end_time > 200: 
            raise ValueError('end_time: 200 Myr is already an overkill.')
        self.dz = z_min/nz 
        self.nz = nz
        self.alpha_g = alpha_g
        self.end_time = end_time
        self.depth_melt = depth_melt
        self.option_1D_solve = option_1D_solve
        self.dt = dt
        self.recalculate = recalculate
        self.van_keken = van_keken
        self.c_age_plate = c_age_plate
        self.c_age_var = np.array(c_age_var, dtype=float64)
        self.z = np.zeros(nz, dtype=float64)
        self.LHS = np.zeros(nz, dtype=float64)
        self.LHS_var = np.zeros((nz, t_res), dtype=float64)
        self.t_res_vec = np.zeros((t_res), dtype=float64)
        self.flag = np.zeros(nz, dtype=int32)
        self.d_RHS = d_RHS
        


    
    
