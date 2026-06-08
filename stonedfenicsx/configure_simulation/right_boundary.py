"""Steady state diffusion problem for a boundary condition on the right boundary and switch to half-space cooling model
"""
from stonedfenicsx.package_import import *
from stonedfenicsx.utils import timing_function, print_ph
from stonedfenicsx.numerical_control import NumericalControls
from stonedfenicsx.scal import Scal
import scipy.sparse.linalg as spla
def configure_thermal_properties(T,phases):
    # Placeholder function to configure thermal properties based on temperature and phases
    # In a real implementation, this would involve complex logic based on the material properties
    k = 3.1  # Thermal conductivity in W/(m*K)
    H = 0.0  # Internal heat generation (W/m^3)
    return k, H

def assembly_matrix(A, k, dx, N):
    
    for i in range(N):
        if i == 0:
            A[i, i] = 1.0  # Dirichlet boundary condition at the left boundary
        elif i == N - 1:
            A[i, i] = 1.0  # Dirichlet boundary condition at the right boundary
        else:
            A[i, i - 1] = k / dx**2
            A[i, i] = -2 * k / dx**2
            A[i, i + 1] = k / dx**2
    
    
    
    return A 

def solve_diffusion_steady_state(k, H, dx, Ttop, Tmax):
    
    
    # Set up the coefficient matrix A and the right-hand side vector b for the finite difference method
    N = len(k)
    A = np.lil_matrix((N, N))
    b = np.zeros(N)
    A = Assembly_matrix(A,k, dx, N)
    b = Assembly_vector(H, dx, N, Ttop, Tmax)
    # Solve the linear system A * T = b
    T = spla.spsolve(A.tocsr(), b)
    return T
    




def solve_1D_steady_state_diffusion():
    
    # Define parameters
    Tmax = 1350.0 + 273.15  # Maximum temperature in Kelvin
    Ttop = 0.0 + 273.15  # Surface temperature in Kelvin
    g = 9.81  # Gravitational acceleration in m/s^2
    L = 100e3  # Length of the domain in meters
    N = 1000  # Number of grid points
    k = 3.1 # Thermal conductivity in W/(m*K)
    x = -np.linspace(0, L, N)  # Spatial grid
    dx = x[1] - x[0]  # Grid spacing
    # Initial guess for temperature distribution (linear profile)
    T_i = np.linspace(Ttop, Tmax, N)
    # Iteratively solve the steady-state diffusion equation
    k, H = configure_thermal_properties(T_i,phases)


    while res < 1e-6:
        
        k, _ = configure_thermal_properties(T_i)
        
        T_new = solve_diffusion_steady_state(k, H ,dx, Ttop, Tmax)
        
        res = np.linalg.norm(T_new - T_i) / np.linalg.norm(T_i)
        
        T_i = T_new
        
        
        
        
        
    
    
    
    
    return T



def right_boundary_LHS():
    
    right_boundary = 'Continental' # or 'continental'
    if right_boundary == 'oceanic': 
        # call the LHS script for the oceanic cooling place holder
        pass 
    else: 
        T = solve_1D_steady_state_diffusion()
    

    return T 




if __name__ == "__main__":
    
    from pathlib import Path
    from stonedfenicsx.numerical_control import NumericalControls,IOControls 
    from stonedfenicsx.create_mesh.aux_create_mesh import Geom_input 
    from stonedfenicsx.scal import Scal
    from stonedfenicsx.utils import Input,print_ph, parse_input
    from stonedfenicsx.create_mesh.create_mesh import create_mesh
    from stonedfenicsx.Stoned_fenicx import fill_geometrical_input
    import sys
    import numpy as np
    from numpy.typing import NDArray
    import os
    from stonedfenicsx.package_import import *
    import argparse



    # Go to the global input file 
    p = Path(__file__).parents[1]
    # Read and parse the input file
    path_input = f"{p}/input.yaml"
    
    inp,Ph = parse_input(path_input)
    
    alpha_nameC = 'Crust'#'Constant'#
    alpha_nameM = 'Mantle'#'Mantle'
    density_nameC = 'PT'#'PT'
    density_nameM = 'PT'#'PT'
    capacity_nameM = 'Bermann_Aranovich_Fo_Fa_0_1'#'Bermann_Aranovich_Fo_Fa_0_1'
    capacity_nameC = 'Oceanic_Crust'#'Oceanic_Crust'
    conductivity_nameM = 'Mantle'#'Mantle'
    conductivity_nameC = 'Oceanic_Crust'#'Oceanic_Crust'
    rho0_M = 3300.0
    rho0_C = 2800.0
    H_r_LC = 0.27e-6
    H_r_UC = 1.5e-6
    H_r_M  = 0.06e-6
    radio_flag = 1 

    Ph.subducting_plate_mantle.rho0 = rho0_M
    Ph.subducting_plate_mantle.name_capacity = capacity_nameM
    Ph.subducting_plate_mantle.name_conductivity = conductivity_nameM
    Ph.subducting_plate_mantle.name_alpha = alpha_nameM
    Ph.subducting_plate_mantle.name_density = density_nameM
    Ph.subducting_plate_mantle.radio_flag = radio_flag
    Ph.subducting_plate_mantle.Hr = H_r_M
    Ph.subducting_plate_mantle.k = 2.5 


    Ph.oceanic_crust.rho0 = rho0_C
    Ph.oceanic_crust.name_capacity = capacity_nameC
    Ph.oceanic_crust.name_conductivity = conductivity_nameC
    Ph.oceanic_crust.name_alpha = alpha_nameC
    Ph.oceanic_crust.name_density = density_nameC
    Ph.oceanic_crust.radio_flag = radio_flag
    Ph.oceanic_crust.Hr = H_r_LC 
    Ph.oceanic_crust.k = 1.5 

    Ph.wedge_mantle.name_diffusion = 'Constant'
    Ph.wedge_mantle.name_dislocation = 'Constant'
    Ph.wedge_mantle.rho0 = rho0_M
    Ph.wedge_mantle.name_capacity = capacity_nameM 
    Ph.wedge_mantle.name_conductivity = conductivity_nameM
    Ph.wedge_mantle.name_alpha = alpha_nameM
    Ph.wedge_mantle.name_density = density_nameM
    Ph.wedge_mantle.radio_flag = radio_flag
    Ph.wedge_mantle.Hr = H_r_M 
    Ph.wedge_mantle.k = 2.5 

    Ph.overriding_mantle.rho0 = rho0_M 
    Ph.overriding_mantle.name_capacity = capacity_nameM
    Ph.overriding_mantle.name_conductivity = conductivity_nameM
    Ph.overriding_mantle.name_alpha = alpha_nameM
    Ph.overriding_mantle.name_density = density_nameM
    Ph.overriding_mantle.radio_flag = radio_flag
    Ph.overriding_mantle.Hr = H_r_M 
    Ph.overriding_mantle.k = 2.5

    Ph.overriding_upper_crust.rho0 = rho0_C 
    Ph.overriding_upper_crust.name_capacity = capacity_nameC
    Ph.overriding_upper_crust.name_conductivity = conductivity_nameC
    Ph.overriding_upper_crust.name_alpha = alpha_nameC
    Ph.overriding_upper_crust.name_density = density_nameC
    Ph.overriding_upper_crust.radio_flag = radio_flag
    Ph.overriding_upper_crust.Hr = H_r_UC 
    Ph.overriding_upper_crust.k = 1.5 

    Ph.overriding_lower_crust.rho0 = rho0_C 
    Ph.overriding_lower_crust.name_capacity = capacity_nameC
    Ph.overriding_lower_crust.name_conductivity = conductivity_nameC
    Ph.overriding_lower_crust.name_alpha = alpha_nameC
    Ph.overriding_lower_crust.name_density = density_nameC
    Ph.overriding_lower_crust.radio_flag = radio_flag
    Ph.overriding_lower_crust.Hr = H_r_LC 
    Ph.overriding_lower_crust.k = 2.0 

    g_input = stonedfenicsx.stoned_fenicx.fill_geometrical_input(Ph,inp)
    Pdb = 



    T = right_boundary_LHS()
    
    
    
    
    
    









