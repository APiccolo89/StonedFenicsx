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
    ('decoupling',int64), # 1 decoupled, 0 coupled
]

@jitclass(spec)
class NumericalControls:
    def __init__(self,
                 it_max=20,
                 tol=1e-8,
                 Ttop=0.0,
                 Tmax=1300,
                 g=9.81, 
                 time_max = 20, 
                 pressure_scaling=1e22/1000e3,
                 slab_age=0.0,
                 v_s = np.array([5.0,0.0], dtype=np.float64),
                 steady_state = 1,
                 relax = 0.6,
                 time_dependent_v = 0,
                 slab_bc = 1, # BC: 0 -> pipe like , 1 moving wall slab
                 decoupling = 0,
                 tol_innerPic = 1e-4,
                 tol_innerNew = 1e-7):  

        # Direct initialization of class attributes
        self.it_max           = it_max 
        self.tol              = tol 
        self.relax            = relax
        self.Tmax             = Tmax + 273.15
        self.Ttop             = Ttop + 273.15
        self.g                = g 
        self.v_s              = v_s  # Convert cm/yr to m/s
        self.slab_age         = slab_age
        self.time_max         = time_max
        self.time_dependent_v = time_dependent_v
        self.steady_state     = steady_state
        self.slab_bc          = 1 # 1 moving wall, 0 pipe-like slab 
        self.decoupling       = decoupling # 1 decoupled, 0 coupled
        self.tol_innerPic    = tol_innerPic
        self.tol_innerNew    = tol_innerNew
    

class IOControls():
    def __init__(self, test_name='', path_save='', sname=''):
        self.test_name = test_name
        self.path_save = path_save
        self.sname = sname
        self.path_test = os.path.join(self.path_save,self.test_name)

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
        van_keken=0,          # benchmark flag
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
        


    
    


main_bc_spec = [('Top_t',int32),
            ('Top_v',float64[:]),
            ('Bot_t',int32),
            ('Bot_v',float64[:]),
            ('Lef_t',int32),
            ('Lef_v',float64[:]),
            ('Rig_t',int32),
            ('Rig_v',float64[:]),
            ('Fea_on',int32)]

@jitclass(main_bc_spec)
class main_bc:
     def __init__(self,
                  Top_t = 0,
                  Top_v = np.array([273.15,0.0],dtype=np.float64),
                  Bot_t = 0,
                  Bot_v = np.array([1573.15,0.0],dtype=np.float64),
                  Lef_t = 1,
                  Lef_v = np.array([-1.0,0.0],dtype=np.float64),
                  Rig_t = 1,
                  Rig_v = np.array([-1.0,0.0],dtype=np.float64),
                  Fea_on = 1,
                  ): 
        self.Top_t = Top_t
        self.Top_v = Top_v
        self.Bot_t = Bot_t
        self.Bot_v = Bot_v
        self.Lef_t = Lef_t
        self.Lef_v = Lef_v
        self.Rig_t = Rig_t
        self.Rig_v = Rig_v
        self.Fea_on = Fea_on

@dataclass
class bc_controls():
    Stokes: main_bc
    Energy: main_bc
    def __init__(self,
                 BC_STOKES,
                 BC_ENERGY):

        self.energy_dic = {'Isothermal':0,
                               'NoFlux':1,
                               'Open':2,
                               'Gaussian':3}
        self.stokes_dic = {'DoNothing':0,
                            'NoSlip':1,
                            'FreeSlip':2,
                            'Traction_lit':3}
        self.Energy = self.fill_class(BC_ENERGY,0)
        self.Stokes = self.fill_class(BC_STOKES,1)
    def fill_class(self,BC,type):
        if type == 0:
            print(':::: Filling Energy Boundary condition specification')
            dict_type = self.energy_dic
        else: 
            print(':::: Filling Stokes Boundary condition specification')
            dict_type = self.stokes_dic

        BC_com = main_bc()

        Top = BC[0][:]
        BC_com.Top_t = dict_type[Top[1]]
        BC_com.Top_v = np.array(Top[2][:])
        Bot = BC[1][:]
        BC_com.Bot_t = dict_type[Bot[1]]
        BC_com.Bot_v = Bot[2][:]
        Lef = BC[2][:]
        BC_com.Lef_t = dict_type[Lef[1]]
        BC_com.Lef_v = np.array(Lef[2][:])
        Rig = BC[3][:]
        BC_com.Rig_t = dict_type[Rig[1]]
        BC_com.Rig_v = np.array(Rig[2][:])
        if BC[4][0] == 'Feature_on':
            BC_com.Fea_on = 1
        else: 
            BC_com.Fea_on = 0 
        
        print(':::: Fertig ::: : .')

        return BC_com


