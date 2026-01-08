import basix.ufl
from src.utils import Phase 


#---------------------------------------------------------------------------------------------------------
# Geometric input parameters: 
x                 = [0, 660e3]               # main grid coordinate
y                 = [-600e3,0.0]   
slab_tk           = 130e3                   # slab thickness
cr                = 30e3              # crust 
ocr               = 7e3             # oceanic crust
lit_mt            = 20e3          # lithosperic mantle  
lc                = 0.3              # lower crust ratio 
wc                = 2.0e3              # weak zone 
lt_d              = (cr+lit_mt)     # total lithosphere thickness
lab_d             = 100e3 
decoupling        = 80e3      # decoupling depth -> i.e. where the weak zone is prolonged 
resolution_normal = 2.0e3  # To Do
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
model_shear       = 'SelfConsistent' # 'SelfConsistent
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
