from stonedfenicsx.numerical_control import NumericalControls,IOControls 
from stonedfenicsx.create_mesh.aux_create_mesh import Geom_input 
from stonedfenicsx.scal import Scal
from stonedfenicsx.utils import Input,print_ph
from stonedfenicsx.create_mesh.create_mesh import create_mesh
from stonedfenicsx.Stoned_fenicx import fill_geometrical_input
import sys
import numpy as np
from numpy.typing import NDArray
import os
from stonedfenicsx.package_import import *
import h5py as h5
# Global flag to decide wether or not to remove the results -> debug reason. 
DEBUG = False
#-------------------------------------------------------------------------------
def perform_test(option_viscous:int=2
                 ,direct:str='Direct'
                 ,rtol:int =0)->None:
    
    
    from stonedfenicsx.utils import parse_input, time_the_time, timing
    from stonedfenicsx.Stoned_fenicx import StonedFenicsx
    
    time_A = timing.time()

    path_test = os.path.dirname(os.path.realpath(__file__))
    
    path_input = f"{path_test}/input_tests.yaml"

    inp,Ph = parse_input(path_input)
    # option for the benchmark

    # Create input data - Input is a class populated by default dataset
    # A flag that generate the geometry of the benchmark
    van_keken = 1
    # The input path for saving the results
    inp.path_test = f'{path_test}/Tests_Van_keken'

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
    inp.stokes_solver_type = direct
    print_ph('Starting the benchmark tests with different options')

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

    if option_viscous == 0:
        name_diffusion = 'Constant'
        name_dislocation = 'Constant'              
    elif option_viscous == 1: 
        name_diffusion = 'Van_Keken_diff'
        name_dislocation = 'Constant'       
    elif option_viscous == 2: 
        name_diffusion = 'Van_Keken_diff'
        name_dislocation = 'Van_Keken_disl'                 
    
    if rtol == 0: 
        inp.iterative_solver_tol = 1e-10 
    if rtol == 1: 
        inp.iterative_solver_tol = 1e-9 
    if rtol == 2: 
        inp.iterative_solver_tol = 1e-8 
    if rtol == 3: 
        inp.iterative_solver_tol = 1e-11 
    if rtol == 4: 
        inp.iterative_solver_tol = 1e-7
    if rtol == 5: 
        inp.iterative_solver_tol = 1e-6

        
    

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


    inp.sname = f'T_viscous{option_viscous}_direct{direct}_rtol{rtol}'

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
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------   
def test_composite_direct()->None:
    # Test Van Keken
    perform_test(2, 'Direct')


def test_composite_iterative(rel_tol:int=0)->None:
    perform_test(2,'Iterative',rel_tol)
    
def read_data_base(): 
    file_h5 = f'{os.path.dirname(os.path.realpath(__file__))}/Tests_Van_keken/benchmark_van_keken.h5'

    def extract_information(solver:str, rtol:int)->tuple[NDArray,NDArray,NDArray,NDArray,float,float]:
        name_file = f'T_viscous2_direct{solver}_rtol{rtol}'
        
        with h5.File(file_h5) as f: 
            RMSv = np.array(f[f'{name_file}/RMSv'])
            RMST = np.array(f[f'{name_file}/RMST'])
            mv = np.array(f[f'{name_file}/minVel'])
            Mv = np.array(f[f'{name_file}/maxVel'])
            L_11_11 = np.array(f[f'{name_file}/T_11_11'])
            if rtol == 0: 
                r = 1e-10 
            if rtol == 1: 
                r = 1e-9 
            if rtol == 2: 
                r = 1e-8 
            if rtol == 3: 
                r = 1e-11
            if solver == 'Direct': 
                r = 1e-22  
        return RMSv, RMST, mv, Mv, L_11_11, r 

    def plot_iterative_series(data,r_v,label):
        import matplotlib.pyplot as plt 
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        fig = plt.figure()
        ax = fig.gca()
        for i in range(len(r_v)):
            ax.plot(range(len(data[i])),data[i],c=colors[i],linewidth=1.0,label = f'tol = {np.log10(r_v[i])}')
        plt.legend()
        ax.set_xlabel(r'$it_{outer}$')
        ax.set_ylabel(r'%s'%label)
        ax.set_yscale('log')
        ax.grid(visible='True',which='both',axis='both', color='k',linewidth=0.3,alpha=0.5,linestyle=':')
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color = 'k'
        plt.show()
        
    def scatter_plot(r_v,L): 
        import matplotlib.pyplot as plt 
        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(r_v,L)
        ax.set_xscale('log')
        ax.set_xlabel(r'$r_t$')
        ax.set_ylabel(r'$L_{11}$')
        ax.grid(visible='True',which='both',axis='both', color='k',linewidth=0.3,alpha=0.5,linestyle=':')
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color = 'k'

        plt.show()
        
        
        
    direct_RMSv, direct_RMST, direct_mv, direct_MV, direct_L11, direct_r = extract_information('Direct', 0)
    I0_RMSv, I0_RMST, I0_mv, I0_MV, I0_L11, I0_r = extract_information('Iterative', 0)
    I1_RMSv, I1_RMST, I1_mv, I1_MV, I1_L11, I1_r = extract_information('Iterative', 1)
    I2_RMSv, I2_RMST, I2_mv, I2_MV, I2_L11, I2_r = extract_information('Iterative', 2)
    I3_RMSv, I3_RMST, I3_mv, I3_MV, I3_L11, I3_r = extract_information('Iterative', 3)
    I4_RMSv, I4_RMST, I4_mv, I4_MV, I4_L11, I4_r = extract_information('Iterative', 4)
    I5_RMSv, I5_RMST, I5_mv, I5_MV, I5_L11, I5_r = extract_information('Iterative', 5)

    Data_0 = [direct_RMSv,I0_RMSv,I1_RMSv,I2_RMSv,I3_RMSv,I4_RMSv,I5_RMSv]
    Data_1 = [direct_RMST,I0_RMST,I1_RMST,I2_RMST,I3_RMST,I4_RMST,I5_RMSv]
    Data_3 = [direct_MV, I0_MV, I1_MV, I2_MV, I3_MV,I4_MV,I5_MV]
    d_D0 = ([I0_RMSv,I1_RMSv,I2_RMSv,I3_RMSv,I4_RMSv,I5_RMSv]-direct_RMSv)/direct_RMSv
    d_D1 = ([I0_RMST,I1_RMST,I2_RMST,I3_RMST,I4_RMST, I5_RMST]-direct_RMST)/direct_RMST
    
    L11 = [direct_L11, I0_L11, I1_L11, I2_L11, I3_L11,I4_L11,I5_L11]
    dL11 = ([I0_L11, I1_L11, I2_L11, I3_L11,I4_L11,I5_L11]-direct_L11)/direct_L11

    r_v = [direct_r, I0_r, I1_r, I2_r, I3_r,I4_r,I5_r]
    
    plot_iterative_series(Data_0,r_v,'$RMS_v$')
    plot_iterative_series(np.abs(d_D0),r_v[1:],r'$\Delta RMS_v$')
    plot_iterative_series(Data_1,r_v,'$RMS_T$')
    plot_iterative_series(np.abs(d_D1),r_v[1:],r'$\Delta RMS_T$')
    plot_iterative_series(Data_3,r_v, '$max(u)$')
    scatter_plot(r_v,L11)
    scatter_plot(r_v[1:],dL11)    
    



    
    


#-------------------------------------------------------------------------------
if __name__ == '__main__': 
    
    DEBUG = True
    
    #test_composite_direct()
    
    #test_composite_iterative(0)
    
    #test_composite_iterative(1)
    
    #test_composite_iterative(2)
    
    #test_composite_iterative(3)
    
    test_composite_iterative(4)
    
    test_composite_iterative(5)
    
    
    read_data_base()
    
#---------------------------------------------------------------------------------
