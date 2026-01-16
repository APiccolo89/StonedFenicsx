from src.utils import Input,Phase,Ph_input
from src.Stoned_fenicx import StonedFenicsx

# Create input data 
inp = Input()

van_keken = 1 

option_van_keken = [0,1,2]

inp.path_test = '../Results/Tests_Van_keken'

inp.alpha_nameC = 'Crust'
inp.alpha_nameM = 'Mantle'
inp.density_nameM = 'PT'
inp.density_nameC = 'PT'
inp.capacity_nameM = 'Bermann_Aranovich_Fo_Fa_0_1'
inp.capacity_nameC = 'Oceanic_Crust'
inp.conductivity_nameC = 'Oceanic_Crust'
inp.conductivity_nameM = 'Mantle'

inp.adiabatic_heating = 1 

for i in range(len(option_van_keken)):
    inp.sname = 'Test_VK_option_AD_SC_%d'%option_van_keken[i]
    # Initialise the input
    inp.van_keken      = van_keken
    inp.van_keken_case = option_van_keken[i]
    
    rho0_M  = inp.rho0_M
    rho0_C  = inp.rho0_C 
    
    alpha_M =  inp.alpha_nameM
    alpha_C = inp.alpha_nameC
    
    density_nameM = inp.density_nameM
    density_nameC = inp.density_nameC 
    
    capacity_nameM = inp.capacity_nameM
    capacity_nameC = inp.capacity_nameC 
    
    alpha_nameM  = inp.alpha_nameM
    alpha_nameC  = inp.alpha_nameC 
    
    conductivity_nameM = inp.conductivity_nameM
    conductivity_nameC = inp.conductivity_nameC
    
    radio_flag = 0 
    
    
    # Set the phase
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
    Phase3.name_diffusion = 'Van_Keken_diff'
    Phase3.name_dislocation = 'Van_Keken_disl'
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
    
    if van_keken == 1: 
        inp.cr          = 0.0 
        inp.lc          = 0.0 
        inp.ocr         = 0.0
        inp.lt_d        = 50e3 
        inp.lit_mt      = 50e3 
        inp.lab_d       = inp.lit_mt
        inp.decoupling  = 50e3 
        inp.Tmax        = 1300.0
        if option_van_keken[i] == 0: 
            Phase3.name_diffusion   = 'Constant'
            Phase3.name_dislocation = 'Constant'
        elif option_van_keken[i] ==1: 
            Phase3.name_diffusion   = 'Van_Keken_diff'
            Phase3.name_dislocation = 'Constant'  
        else:      
            Phase3.name_diffusion   = 'Van_Keken_diff'
            Phase3.name_dislocation = 'Van_Keken_disl'  
    
    Ph_inp = Ph_input()
    Ph_inp.Phase1 = Phase1
    Ph_inp.Phase2 = Phase2
    Ph_inp.Phase3 = Phase3
    Ph_inp.Phase4 = Phase4
    Ph_inp.Phase5 = Phase5
    Ph_inp.Phase6 = Phase6
    Ph_inp.Phase7 = Phase7
    StonedFenicsx(inp,Ph_inp)
    print('Finished the first test============================================================')


    



