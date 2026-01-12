import basix.ufl
from src.utils import Phase, Ph_input


#---------------------------------------------------------------------------------------------------------
# Geometric input parameters: 
#---------------------------------------------------------------------------------------------------------
x                 = [0, 660e3]               # main grid coordinate max X is changed accordingly to the slab geometry
y                 = [-600e3,0.0]             # Initial input
slab_tk           = 130e3                   # slab thickness
cr                = 30e3              # crust 
ocr               = 7e3             # oceanic crust
lit_mt            = 20e3          # lithosperic mantle  
lc                = 0.3              # lower crust ratio 
wc                = 2.0e3          # width of the weak zone
lt_d              = (cr+lit_mt)     # total lithosphere thicknes [and depth of the no slip boundary conditiopn]
lab_d             = 100e3   # lithosphere-asthenosphere boundary depth -> useful for tuning the initial temperature profile
decoupling        = 80e3      # decoupling depth -> i.e. where the weak zone is prolonged 
resolution_normal = 2.0e3  # Normal resolution
transition        = 10e3    # Parameter that controls the transition depth of velocity in the decoupling zone
#---------------------------------------------------------------------------------------------------------
# Numerical Controls 
#---------------------------------------------------------------------------------------------------------
it_max            = 20 
tol               = 1e-5 
relax             = 0.9
Tmax              = 1333.0
Ttop              = 0.0 
g                 = 9.81 
v_s               = [5.0,0.0]  # Convert cm/yr to m/s
slab_age          = 50
time_max          = 2
time_dependent_v  = 0
steady_state      = 1
slab_bc           = 1 # 1 moving wall, 0 pipe-like slab 
tol_innerPic      = 1e-2
tol_innerNew      = 1e-5
van_keken_case    = 2 # 1 Van Keken benchmark, When these flag are activated -> default value in all phases -> 3300  
                    # 2 diffusion only, 
                    # 3 composite 
                    # 4 Iterative 
decoupling_ctrl   = 1
model_shear       = 'NoShear' # 'SelfConsistent
phase_wz          = 7
time_dependent    = 1 
dt_sim            = 15000/1e6 # in Myr

friction_angle   = 1.0 
#---------------------------------------------------------------------------------------------------------
# input/output control
#---------------------------------------------------------------------------------------------------------
test_name = 'Output'
sname = test_name
#---------------------------------------------------------------------------------------------------------
# Scaling parameters
#---------------------------------------------------------------------------------------------------------
L       = 600e3          # length scale [m]
stress  = 1e9          # stress scale [Pa]
eta     = 1e21         # viscosity scale [Pa.s]
Temp    = 1333.0       # temperature scale [K]
#---------------------------------------------------------------------------------------------------------
# Left boundary condition 
#---------------------------------------------------------------------------------------------------------
nz = 108
end_time = 180.0
dt = 0.001
recalculate = 1
van_keken = 0
c_age_plate = 50.0 
flag_radio = 0.0 
#---------------------------------------------------------------------------------------------------------
# Phase properties
# ---------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------
element_p = basix.ufl.element("Lagrange","triangle", 1) 
element_PT = basix.ufl.element("Lagrange","triangle",2)
element_V = basix.ufl.element("Lagrange","triangle",2,shape=(2,))
#---------------------------------------------------------------------------------------------------------
Phase1 = Phase()
Phase1.name_phase = 'Mantle_Slab'
Phase1.rho0 = 3300.0
Phase1.name_alpha = 'Mantle'
Phase1.name_density = 'PT'
Phase1.name_capacity = 'Bermann_Aranovich_Fo_Fa_0_1'
Phase1.name_conductivity = 'Mantle'
Phase1.radio_flag = 1.0

Phase2 = Phase()
Phase2.name_phase = 'Crust_Slab'
Phase2.rho0 = 2900.0
Phase2.name_alpha = 'Crust'
Phase2.name_density = 'PT'
Phase2.name_capacity = 'Oceanic_Crust'

Phase3 = Phase()
Phase3.name_phase = 'Mantle_WG'
Phase3.rho0 = 3300.0
Phase3.name_diffusion = 'Van_Keken_diff'
Phase3.name_dislocation = 'Van_Keken_disl'
Phase3.name_alpha = 'Mantle'
Phase3.name_density = 'PT'
Phase3.name_capacity = 'Bermann_Aranovich_Fo_Fa_0_1'
Phase3.name_conductivity = 'Mantle'
Phase3.radio_flag = 1.0 

Phase4 = Phase()
Phase4.name_phase = 'Mantle_Lithosphere'
Phase4.rho0 = 3300.0
Phase4.name_alpha = 'Mantle'
Phase4.name_density = 'PT'
Phase4.name_capacity = 'Bermann_Aranovich_Fo_Fa_0_1'
Phase4.name_conductivity = 'Mantle'
Phase4.radio_flag = 1.0 

Phase5 = Phase()
Phase5.name_phase = 'Crust_Lithosphere'
Phase5.rho0 = 2800.0
Phase5.name_alpha = 'Crust'
Phase5.name_density = 'PT'
Phase5.name_capacity = 'Oceanic_Crust'
Phase5.name_conductivity = 'Oceanic_Crust'
Phase5.radio_flag = 1.0

Phase6 = Phase()
Phase6.name_phase = 'Lower_Crust_Lithosphere'
Phase6.rho0 = 2800.0
Phase6.name_alpha = 'Crust'
Phase6.name_density = 'PT'
Phase6.name_capacity = 'Oceanic_Crust'
Phase6.name_conductivity = 'Oceanic_Crust'
Phase6.radio_flag = 1.0

Phase7 = Phase()
Phase7.name_phase = 'Weak_Zone'
Phase7.name_diffusion = 'Hirth_Wet_Olivine_diff'
Phase7.name_dislocation = 'Hirth_Wet_Olivine_disl'

Ph_input = Ph_input()
Ph_input.Phase1 = Phase1
Ph_input.Phase2 = Phase2
Ph_input.Phase3 = Phase3
Ph_input.Phase4 = Phase4
Ph_input.Phase5 = Phase5
Ph_input.Phase6 = Phase6
Ph_input.Phase7 = Phase7