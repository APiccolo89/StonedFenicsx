import basix.ufl

class Phase():
    pass 
#---------------------------------------------------------------------------------------------------------
# Geometric input parameters: 
x                 = [0, 660e3]               # main grid coordinate
y                 = [-600e3,0.0]   
slab_tk           = 130e3                   # slab thickness
cr                = 35e3              # crust 
ocr               = 6e3             # oceanic crust
lit_mt            = 15e3          # lithosperic mantle  
lc                = 0.5              # lower crust ratio 
wc                = 2.0e3              # weak zone 
lt_d              = (cr+lit_mt)     # total lithosphere thickness
decoupling        = 100e3      # decoupling depth -> i.e. where the weak zone is prolonged 
resolution_normal = 2.0e3  # To Do
#---------------------------------------------------------------------------------------------------------
# Numerical Controls 
#---------------------------------------------------------------------------------------------------------
it_max           = 20 
tol              = 1e-5 
relax            = 0.9
Tmax             = 1350.0
Ttop             = 0.0 
g                = 9.81 
v_s              = [5.0,0.0]  # Convert cm/yr to m/s
slab_age         = 80e6 
time_max         = 20e6
time_dependent_v = 0
steady_state     = 1
slab_bc          = 1 # 1 moving wall, 0 pipe-like slab 
tol_innerPic    = 1e-4
tol_innerNew    = 1e-7
case_VanKeken   = 0 # 1 Van Keken benchmark, 2 diffusion only, 3 composite 
decoupling_ctrl = 1 
#---------------------------------------------------------------------------------------------------------
# input/output control
#---------------------------------------------------------------------------------------------------------
test_name = 'Experiment_1'
path_save = '/Users/wlnw570/Work/Leeds/Output/Stonedphoenix/Benchmark'
sname = test_name
#---------------------------------------------------------------------------------------------------------
# Scaling parameters
#---------------------------------------------------------------------------------------------------------
L       = 600e3          # length scale [m]
stress  = 1e9          # stress scale [Pa]
eta     = 1e21         # viscosity scale [Pa.s]
Temp    = 1300.0       # temperature scale [K]
#---------------------------------------------------------------------------------------------------------
# Left boundary condition 
#---------------------------------------------------------------------------------------------------------
nz = 108
end_time = 180.0
dt = 0.001
recalculate = 1
van_keken = 0
c_age_plate = 50 
#---------------------------------------------------------------------------------------------------------
# Phase properties
# ---------------------------------------------------------------------------------------------------------
Phase1 = Phase()
Phase1.name = 'Slab_Mantle'
Phase1.rho0 = 3300.0
Phase1.option_rho = 0
Phase1.option_rheology = 0  
Phase1.option_k = 0
Phase1.option_Cp = 0
Phase1.radio    = 0.066e-6
Phase1.eta = 1e22
#---------------------------------------------------------------------------------------------------------
Phase2 = Phase()
Phase2.name = 'Oceanic_Crust'
Phase2.rho0 = 3300.0
Phase2.option_rho = 0
Phase2.option_rheology = 0  
Phase2.option_k = 0
Phase2.option_Cp = 0
Phase2.radio    = 0.066e-6
Phase2.eta = 1e22
#---------------------------------------------------------------------------------------------------------
Phase3 = Phase()
Phase3.name = 'Wedge'
Phase3.rho0 = 3300.0
Phase3.option_rho = 0
Phase3.option_rheology = 3  
Phase3.option_k = 0
Phase3.option_Cp = 0
Phase3.radio    = 0.066e-6

Phase3.name_diffusion = 'Van_Keken_diff'
Phase3.name_dislocation = 'Van_Keken_disl'
Phase3.eta = 1e19
#---------------------------------------------------------------------------------------------------------
Phase4 = Phase()
Phase4.name = 'Continental_Mantle'
Phase4.rho0 = 3300.0
Phase4.option_rho = 0
Phase4.option_rheology = 0  
Phase4.option_k = 0
Phase4.radio    = 0.066e-6
Phase4.option_Cp = 0
Phase4.eta = 1e22
#---------------------------------------------------------------------------------------------------------
Phase5 = Phase()
Phase5.name = 'Continental_Crust'
Phase5.rho0 = 3300.0
Phase5.option_rho = 0
Phase5.option_rheology = 0  
Phase5.option_k = 0
Phase5.option_Cp = 0
Phase5.eta = 1e22
Phase5.radio    = 0.066e-6

#---------------------------------------------------------------------------------------------------------
Phase6 = Phase()
Phase6.name = 'Lower_Crust'
Phase6.rho0 = 3300.0
Phase6.option_rho = 0
Phase6.option_rheology = 0
Phase6.option_k = 0
Phase6.option_Cp = 0
Phase6.eta = 1e22
Phase6.radio    = 0.066e-6

#---------------------------------------------------------------------------------------------------------
element_p = basix.ufl.element("Lagrange","triangle", 1) 
element_PT = basix.ufl.element("Lagrange","triangle",2)
element_V = basix.ufl.element("Lagrange","triangle",2,shape=(2,))
#---------------------------------------------------------------------------------------------------------