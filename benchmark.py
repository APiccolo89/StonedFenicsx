from src.utils import Input,Phase,Ph_input,print_ph,time_the_time
import time as timethis
from src.Stoned_fenicx import StonedFenicsx
# Create the script for the benchmark tests


# option for the benchmark
option_thermal   = [3]
option_adiabatic = [0]
option_viscous   = [0,1,2]
self_con         = [0,1]


# Create input data - Input is a class populated by default dataset
inp = Input()
# A flag that generate the geometry of the benchmark
van_keken = 1 

# The input path for saving the results
inp.path_test = '../Results/Tests_Van_keken'

# Geometrical input
inp.cr          = 0.0 # Overriding crust 
inp.lc          = 0.3 # relative amount of lower crust
inp.ocr         = 6.0e3 # Crustal thickness
inp.lt_d        = 50e3 # No slip boundary condition depth
inp.lit_mt      = 50e3 # Lithospheric mantle depth 
inp.lab_d       = inp.lit_mt # depth of the lab 
inp.decoupling  = 50e3  # decoupling depth
inp.Tmax        = 1300.0 # mantle potential temperature
#inp.model_shear = 'SelfConsistent'

print_ph('Starting the benchmark tests with different options')

time_in = timethis.time()

for i in range(len(option_thermal)):
    for j in range(len(option_adiabatic)):
        for k in range(len(option_viscous)):
            for l in range(len(self_con)):
                time_A = timethis.time()
                radio_flag = 1 

                if option_thermal[i] == 0: 
                    alpha_nameC = 'Constant'
                    alpha_nameM = 'Constant'
                    density_nameC   = 'Constant'
                    density_nameM   = 'Constant'
                    capacity_nameM    = 'Constant'
                    capacity_nameC    = 'Constant'
                    conductivity_nameM     = 'Constant'
                    conductivity_nameC     = 'Constant'
                    rho0_M   = 3300.0
                    rho0_C   = 3300.0
                    radio_flag = 0 

                elif option_thermal[i] == 1: 
                    alpha_nameC = 'Mantle'
                    alpha_nameM = 'Mantle'
                    density_nameC   = 'PT'
                    density_nameM   = 'PT'
                    capacity_nameM    = 'Bermann_Aranovich_Fo_Fa_0_1'
                    capacity_nameC    = 'Bermann_Aranovich_Fo_Fa_0_1'
                    conductivity_nameM     = 'Mantle'
                    conductivity_nameC     = 'Mantle'
                    rho0_M   = 3300.0
                    rho0_C   = 3300.0
                elif option_thermal[i] == 2 or option_thermal[i] == 3: 
                    alpha_nameC = 'Crust'
                    alpha_nameM = 'Mantle'
                    density_nameC   = 'PT'
                    density_nameM   = 'PT'
                    capacity_nameM    = 'Bermann_Aranovich_Fo_Fa_0_1'
                    capacity_nameC    = 'Oceanic_Crust'
                    conductivity_nameM     = 'Mantle'
                    conductivity_nameC     = 'Oceanic_Crust'
                    rho0_M   = 3300.0
                    rho0_C   = 2900.0
                    if option_thermal[i] == 3: 
                        inp.cr = 30e3
                        
                        

                if option_adiabatic[j] == 0: 
                    inp.adiabatic_heating = 0 
                elif option_adiabatic[j] == 1: 
                    inp.adiabatic_heating = 1
                elif option_adiabatic[j] == 2: 
                    inp.adiabatic_heating = 2
                    inp.lab_d       = 80e3



                if option_viscous[k] == 0: 
                    name_diffusion   = 'Constant'
                    name_dislocation = 'Constant'              
                elif option_viscous[k] == 1: 
                    name_diffusion   = 'Van_Keken_diff'
                    name_dislocation = 'Constant'    

                elif option_viscous[k] == 2: 
                    name_diffusion   = 'Van_Keken_diff'
                    name_dislocation = 'Van_Keken_disl'              

                if self_con[l]==1: 
                    inp.self_consistent_flag = 0 
                else: 
                    inp.self_consistent_flag = 1 
                inp.sname = 'T_%d_%d_%d_%d'%(option_viscous[k],option_thermal[i],option_adiabatic[j],self_con[l])

                # Initialise the input
                inp.van_keken        = van_keken




                Phase1 = Phase()
                Phase1.name_phase = 'Mantle_Slab'
                Phase1.rho0 = rho0_M
                Phase1.name_alpha = alpha_nameM
                Phase1.name_density = density_nameM
                Phase1.name_capacity = capacity_nameM
                Phase1.name_conductivity = conductivity_nameM
                Phase1.radio_flag = radio_flag

                Phase2 = Phase()
                Phase2.name_phase = 'Crust_Slab'
                Phase2.rho0 = rho0_C
                Phase2.name_alpha = alpha_nameC
                Phase2.name_density = density_nameC
                Phase2.name_conductivity = conductivity_nameC
                Phase2.name_capacity = capacity_nameM
                Phase2.radio_flag = radio_flag

                Phase3 = Phase()
                Phase3.name_phase = 'Mantle_WG'
                Phase3.rho0 = rho0_M
                Phase3.name_diffusion = name_diffusion
                Phase3.name_dislocation = name_dislocation
                Phase3.name_alpha = alpha_nameM
                Phase3.name_density = density_nameM
                Phase3.name_capacity = capacity_nameM
                Phase3.name_conductivity = conductivity_nameM
                Phase3.radio_flag = radio_flag

                Phase4 = Phase()
                Phase4.name_phase = 'Mantle_Lithosphere'
                Phase4.rho0 = rho0_M
                Phase4.name_alpha = alpha_nameM
                Phase4.name_density = density_nameM
                Phase4.name_capacity = capacity_nameM
                Phase4.name_conductivity = conductivity_nameM
                Phase4.radio_flag = radio_flag 

                Phase5 = Phase()
                Phase5.name_phase = 'Crust_Lithosphere'
                Phase5.rho0 = rho0_C
                Phase5.name_alpha = alpha_nameC
                Phase5.name_density = density_nameC
                Phase5.name_capacity = capacity_nameC
                Phase5.name_conductivity = conductivity_nameC
                Phase5.radio_flag = radio_flag

                Phase6 = Phase()
                Phase6.name_phase = 'Lower_Crust_Lithosphere'
                Phase6.rho0 = rho0_C
                Phase6.name_alpha = alpha_nameC
                Phase6.name_density = density_nameC
                Phase6.name_capacity = capacity_nameC
                Phase6.name_conductivity = conductivity_nameC
                Phase6.radio_flag = radio_flag

                Phase7 = Phase()
                Phase7.name_phase = 'Weak_Zone'
                Phase7.name_diffusion = 'Hirth_Wet_Olivine_diff'
                Phase7.name_dislocation = 'Hirth_Wet_Olivine_disl'

                Ph_inp = Ph_input()
                Ph_inp.Phase1 = Phase1
                Ph_inp.Phase2 = Phase2
                Ph_inp.Phase3 = Phase3
                Ph_inp.Phase4 = Phase4
                Ph_inp.Phase5 = Phase5
                Ph_inp.Phase6 = Phase6
                Ph_inp.Phase7 = Phase7

                StonedFenicsx(inp,Ph_inp)

                time_B = timethis.time()
                dt = time_B - time_A
                print('#--------------------------------------------------------------------#')
                if dt > 60.0:
                    m, s = divmod(dt, 60)
                    print(f"{inp.sname} took {m:.2f} min and {s:.2f} sec")
                if dt > 3600.0:
                    m, s = divmod(dt, 60)
                    h, m = divmod(m, 60)
                    print(f"{inp.sname} took {dt/3600:.2f} hr, {m:.2f} min and {s:.2f} sec")
                else:
                    print(f"{inp.sname} took  {dt:.2f} sec")
                print('#--------------------------------------------------------------------#')



time_fin = timethis.time()

dt = time_fin - time_in

if dt > 60.0:
    m, s = divmod(dt, 60)
    print(f" All the experiments took {m:.2f} min and {s:.2f} sec")
if dt > 3600.0:
    m, s = divmod(dt, 60)
    h, m = divmod(m, 60)
    print(f"All the experiments took {dt/3600:.2f} hr, {m:.2f} min and {s:.2f} sec")
else:
    print(f"All the experiments took  {dt:.2f} sec")


    



