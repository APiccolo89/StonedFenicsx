from stonedfenicsx.utils import parse_input,timing, print_ph, timing
from stonedfenicsx.Stoned_fenicx import StonedFenicsx

# Simple script for perfomring the ensemble of benchmarks of VanKeken with all the option avaiable of the code

path_input = "input.yaml"

inp,Ph = parse_input(path_input)


# option for the benchmark
option_thermal = [0,1,2,3]
option_adiabatic = [0]
option_viscous = [0, 1, 2]
self_con = [0, 1]


# Create input data - Input is a class populated by default dataset
# A flag that generate the geometry of the benchmark
van_keken = 1
# The input path for saving the results
inp.path_test = '../Results/Tests_Van_keken'

# Geometrical input
inp.cr = 0.0   # Overriding crust 
inp.lc = 0.3   # relative amount of lower crust
inp.ocr = 6.0e3  # Crustal thickness
inp.lit_mt = 50e3  # Lithospheric mantle depth 
inp.lab_d = inp.lit_mt  # depth of the lab 
inp.decoupling = 50e3  # decoupling depth
inp.ns_depth = 50e3
inp.Tmax = 1300.0  # mantle potential temperature
# inp.model_shear = 'SelfConsistent'
inp.steady_state = 1
print_ph('Starting the benchmark tests with different options')


time_in = timing.time()

for i in range(len(option_thermal)):
    for j in range(len(option_adiabatic)):
        for k in range(len(option_viscous)):
            for m in range(len(self_con)):
                time_A = timing.time()
                radio_flag = 1 

                if option_thermal[i] == 0: 
                    alpha_nameC = 'Constant'
                    alpha_nameM = 'Constant'
                    density_nameC = 'Constant'
                    density_nameM = 'Constant'
                    capacity_nameM = 'Constant'
                    capacity_nameC = 'Constant'
                    conductivity_nameM = 'Constant'
                    conductivity_nameC = 'Constant'
                    rho0_M = 3300.0
                    rho0_C = 3300.0
                    radio_flag = 0 

                elif option_thermal[i] == 1: 
                    alpha_nameC = 'Mantle'
                    alpha_nameM = 'Mantle'
                    density_nameC = 'PT'
                    density_nameM = 'PT'
                    capacity_nameM = 'Bermann_Aranovich_Fo_Fa_0_1'
                    capacity_nameC = 'Bermann_Aranovich_Fo_Fa_0_1'
                    conductivity_nameM = 'Mantle'
                    conductivity_nameC = 'Mantle'
                    rho0_M = 3300.0
                    rho0_C = 3300.0
                    
                elif option_thermal[i] == 2 or option_thermal[i] == 3: 
                    alpha_nameC = 'Crust'
                    alpha_nameM = 'Mantle'
                    density_nameC = 'PT'
                    density_nameM = 'PT'
                    capacity_nameM = 'Bermann_Aranovich_Fo_Fa_0_1'
                    capacity_nameC = 'Oceanic_Crust'
                    conductivity_nameM = 'Mantle'
                    conductivity_nameC = 'Oceanic_Crust'
                    rho0_M = 3300.0
                    rho0_C = 2900.0
                    
                    if option_thermal[i] == 3:
                        inp.cr = 30e3                   

                if option_adiabatic[j] == 0:
                    inp.adiabatic_heating = 0
                elif option_adiabatic[j] == 1:
                    inp.adiabatic_heating = 1
                elif option_adiabatic[j] == 2:
                    inp.adiabatic_heating = 2
                    inp.lab_d = 80e3



                if option_viscous[k] == 0:
                    name_diffusion = 'Constant'
                    name_dislocation = 'Constant'              
                elif option_viscous[k] == 1: 
                    name_diffusion = 'Van_Keken_diff'
                    name_dislocation = 'Constant'    

                elif option_viscous[k] == 2: 
                    name_diffusion = 'Van_Keken_diff'
                    name_dislocation = 'Van_Keken_disl'              

                if self_con[m] ==1:
                    inp.self_consistent_flag = 0
                else:
                    inp.self_consistent_flag = 1
                    
                    
                # Modify the phase with the new data: 
                Ph.subducting_plate_mantle.rho0 = rho0_M
                Ph.subducting_plate_mantle.name_capacity = capacity_nameM
                Ph.subducting_plate_mantle.name_conductivity = conductivity_nameM
                Ph.subducting_plate_mantle.name_alpha = alpha_nameM
                Ph.subducting_plate_mantle.name_density = density_nameM
                Ph.subducting_plate_mantle.radio_flag = radio_flag
                
                
                Ph.oceanic_crust.rho0 = rho0_C
                Ph.oceanic_crust.name_capacity = capacity_nameC
                Ph.oceanic_crust.name_conductivity = conductivity_nameC
                Ph.oceanic_crust.name_alpha = alpha_nameC
                Ph.oceanic_crust.name_density = density_nameC
                Ph.oceanic_crust.radio_flag = radio_flag
                
                Ph.wedge_mantle.name_diffusion = name_diffusion
                Ph.wedge_mantle.name_dislocation = name_dislocation
                Ph.wedge_mantle.rho0 = rho0_M
                Ph.wedge_mantle.name_capacity = capacity_nameM 
                Ph.wedge_mantle.name_conductivity = capacity_nameM
                Ph.wedge_mantle.name_alpha = alpha_nameM
                Ph.wedge_mantle.name_density = density_nameM
                Ph.wedge_mantle.radio_flag = radio_flag
                
                Ph.overriding_mantle.rho0 = rho0_M 
                Ph.overriding_mantle.name_capacity = capacity_nameM
                Ph.overriding_mantle.name_conductivity = conductivity_nameM
                Ph.overriding_mantle.name_alpha = alpha_nameM
                Ph.overriding_mantle.name_density = density_nameM
                Ph.overriding_mantle.radio_flag = radio_flag
                
                Ph.overriding_upper_crust.rho0 = rho0_C 
                Ph.overriding_upper_crust.name_capacity = capacity_nameC
                Ph.overriding_upper_crust.name_conductivity = conductivity_nameC
                Ph.overriding_upper_crust.name_alpha = alpha_nameC
                Ph.overriding_upper_crust.name_density = density_nameC
                Ph.overriding_upper_crust.radio_flag = radio_flag
                
                Ph.overriding_lower_crust.rho0 = rho0_C 
                Ph.overriding_lower_crust.name_capacity = capacity_nameC
                Ph.overriding_lower_crust.name_conductivity = conductivity_nameC
                Ph.overriding_lower_crust.name_alpha = alpha_nameC
                Ph.overriding_lower_crust.name_density = density_nameC
                Ph.overriding_lower_crust.radio_flag = radio_flag
                
                Ph.virtual_weak_zone.name_diffusion = 'Hirth_Wet_Olivine_disl'
                Ph.virtual_weak_zone.name_dislocation = 'Hirth_Wet_Olivine_disl' 
    
                
                inp.sname = 'T_%d_%d_%d_%d'%(option_viscous[k],option_thermal[i],option_adiabatic[j],self_con[m])

                # Initialise the input
                inp.van_keken = van_keken

            
                StonedFenicsx(inp, Ph)

                time_B = timing.time()
                dt = time_B - time_A
                print('#---------------------------------------------------#')
                if dt > 60.0:
                    m, s = divmod(dt, 60)
                    print(f"{inp.sname} took {m:.2f} min and {s:.2f} sec")
                elif dt > 3600.0:
                    m, s = divmod(dt, 60)
                    h, m = divmod(m, 60)
                    print(f"{inp.sname} took {dt/3600:.2f} hr, {m:.2f} min and {s:.2f} sec")
                else:
                    print(f"{inp.sname} took  {dt:.2f} sec")
                print('#---------------------------------------------------#')

time_fin = timing.time()

dt = time_fin - time_in

if dt > 60.0:
    m, s = divmod(dt, 60)
    print(f" All the experiments took {m:.2f} min and {s:.2f} sec")
elif dt > 3600.0:
    m, s = divmod(dt, 60)
    h, m = divmod(m, 60)
    print(f"All the experiments took {dt/3600:.2f} hr, {m:.2f} min and {s:.2f} sec")
else:
    print(f"All the experiments took  {dt:.2f} sec")