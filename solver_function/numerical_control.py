from dataclasses import dataclass,field 
import numpy as np
import os
from numba.experimental import jitclass
from numba import int64, float64,int32, types
from typing import Tuple, List
from typing import Optional
from numba import njit, prange


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
spec = [
    ('option_k', int64),
    ('option_Cp', int64),
    ('option_rho', int64),
    ('recalculate', int64),
    ('rheology', int64),
    ('it_max', int64),
    ('tol', float64),
    ('relax', float64),
    ('eta_max', float64),
    ('eta_def', float64),
    ('eta_min', float64),
    ('Tmax', float64),
    ('Ttop', float64),
    ('g', float64),
    ('pressure_scaling', float64),
    ('slab_age', float64),
    ('R', float64),
    ('yr2s', float64),
    ('cm2m', float64),
    ('v_s', float64[:]),
    ('scal_year',float64),
    ('scal_vel',float64),
    ('b_vk',int64),
    ('b_vk_t',int64),
    ('pressure_bc',int64),
    ('advect_temperature',int64),
    ('dt', float64),
    ('t_max', float64),  # Maximum time in seconds
    ('petsc',int32),
    ('time_dependent_v',int64)# Assuming this is a NumPy array
]

@jitclass(spec)
class NumericalControls:
    def __init__(self,
                 option_k=0,
                 option_Cp=0,
                 option_rho=0, 
                 rheology=0, 
                 it_max=100,
                 tol=1e-8,
                 relax=0.6,
                 eta_max=1e26, 
                 eta_def=1e20, 
                 eta_min=1e16,
                 Ttop=0.0,
                 Tmax=1300,
                 g=9.81, 
                 pressure_scaling=1e22/1000e3,
                 slab_age=0.0,
                 v_s=np.array([5.0,0.0], dtype=np.float64),
                 scal_vel=1e-2/(365.25*24*60*60),
                 scal_year= 365.25*24*60*60,
                 b_vk = 1,
                 pressure_bc = 1,
                 advect_temperature = 1,
                 dt = 0.0,  # Default time step in seconds
                 petsc = 0,
                 time_dependent_v = 1):

        # Direct initialization of class attributes
        self.option_k = option_k
        self.option_Cp = option_Cp
        self.option_rho = option_rho
        self.rheology = rheology
        self.it_max = it_max 
        self.tol = tol 
        self.relax = relax
        self.eta_max = eta_max
        self.eta_def = eta_def
        self.eta_min = eta_min
        self.Tmax = Tmax + 273.15
        self.Ttop = Ttop + 273.15
        self.g = g 
        self.pressure_scaling = pressure_scaling
        self.R = 8.3145
        self.scal_year = scal_year
        self.scal_vel  = scal_vel
        self.v_s = v_s*np.float64(scal_vel)  # Convert cm/yr to m/s
        self.slab_age = slab_age*scal_year 
        self.b_vk = b_vk 
        self.b_vk_t = rheology
        self.pressure_bc = pressure_bc
        self.advect_temperature = advect_temperature
        self.dt = dt 
        self.petsc = petsc
        time_dependent_v = time_dependent_v
        
        








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



        

spec_LHS = [('dz',float64),
            ('nz',int32),
            ('alpha_g',float64),
            ('end_time',float64),
            ('depth_melt',float64),
            ('option_1D_solve',int32),
            ('dt',float64),
            ('recalculate',int32),
            ('van_keken',int32),
            ('z',float64[:]),
            ('LHS',float64[:]),
            ('d_RHS',float64)]

@jitclass(spec_LHS)
class ctrl_LHS:    
    def __init__(self,
            dz = 1e3,
            nz = 108,
            alpha_g = 3e-5,
            end_time = 80e6,
            depth_melt = 0.,
            option_1D_solve = 1,
            dt = 1e3,
            recalculate = 0,
            van_keken = 0,
            d_RHS = -50e3):
        
        self.dz  = dz 
        self.nz  = nz
        self.alpha_g = alpha_g
        self.depth_melt = depth_melt
        self.option_1D_solve = option_1D_solve
        self.end_time = end_time
        self.dt = dt 
        self.recalculate = recalculate
        self.van_keken = van_keken 
        self.LHS       = np.zeros(nz,dtype=float64)
        self.z         = np.zeros(nz,dtype=float64)
        self.d_RHS     = d_RHS

    def _scale_parameters(self,ctrl):
        self.end_time = self.end_time*ctrl.scal_year
        self.dt       = self.dt*ctrl.scal_year
        return self 

    
    


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


