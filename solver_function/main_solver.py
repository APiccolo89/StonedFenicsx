
import numpy                                as np
import time                                 as timing
import os
import reading_mesh.mesh_parser             as mp
import solver_function.bc                   as bc 
import solver_function.solution_fem         as sol
import matplotlib.pyplot                    as plt 

from material_property.phase_db             import PhaseDataBase
from solver_function.numerical_control      import NumericalControls
from solver_function.numerical_control      import ctrl_LHS
from solver_function.numerical_control      import IOControls
from solver_function.numerical_control      import bc_controls
from solver_function.scal                   import Scal

from reading_mesh.initial_setup             import _generate_initial_setup_temperature
from reading_mesh.initial_setup             import generate_phase_field
from visualisation_output.post_process      import shape_function_visual 
from visualisation_output.post_process      import _precompute_shape 
from visualisation_output.visual_fieldstone import triangulation_visualisation
from visualisation_output.post_process      import _output_routines

from reading_mesh.Mesh_c                    import compute_pressure_field_mesh
import copy

import cProfile

def _initial_guess_temperature(M,BC,S,CD,pdb,ctrl_n,lhs,bc_spec,sc):

    min_A = np.min(S.velocities.area)
    v_max = np.max(np.sqrt(S.velocities.u**2+S.velocities.v**2))
    kappa = 1e-6*(sc.T/sc.L**2)
    dt = np.min([np.sqrt(min_A)/(v_max),min_A/(2*kappa)])
    ctrl_n.dt = 0.7*dt
    time = 0.0
    while time < 0.05*(365.25*60*60*24*1e6/sc.T):
       #-- Set the boundary conditions for the energy equation

        BC= bc.set_boundary_condition_energy(M,BC,S.velocities.u,S.velocities.v,ctrl_n,lhs,bc_spec.Energy,sc)

        #-- Solve the energy equation
        S = sol.solve_system_energy(M,S,CD,BC,ctrl_n,pdb,0,sc,True)
        time = time + dt 
        print('Initial guess temperature, time = %.3f'%(time/(365.25*60*60*24*1e6/sc.T))) 


    return S

def compute_slab(time,sc,ctrl): 
    time_max = ctrl.t_max
    dt = 2*ctrl.scal_year*1e6/sc.T
    v0 = 10*ctrl.scal_vel*sc.T/sc.L 
    v1 = 3*ctrl.scal_vel*sc.T/sc.L 
    t = np.linspace(0,time_max,1000)
    v_t = v0 + ((v1-v0)/time_max)*t
    box_vel = np.zeros(len(t),dtype=np.float64)
    time_interval = []
    time_interval.append([0,dt])
    for i in range(int(time_max/dt)):
        if i == 0: 
            a = time_interval[i][0]
            b = time_interval[i][1]
            c = (t>=a)&(t<=b)
        else: 
            time_interval.append([time_interval[i-1][1],time_interval[i-1][1]+dt])
            a = time_interval[i][0]
            b = time_interval[i][1]
            c = (t>a)&(t<=b)


        box_vel[c] = np.mean(v_t[c])
    
    ind_t = np.where(t>=time)
    v_s  = box_vel[ind_t[0][0]]
    return v_s 


def _solv_iField(pdb:PhaseDataBase,ctrl_n:NumericalControls,ioctrl:IOControls,lhs:ctrl_LHS,bc_spec:bc_controls,sc:Scal):
    
    
    debug = False 
    #-- 
    #-- 1. Create the mesh, boundary conditions and solution class
    #-  1.a Create mesh & scale the grid
    #-  1.b Create the boundary conditions
    #-  1.c Create the solution class and precompute the shape functions
    
    #-- 1. Create the mesh
    #- - 1.a Create mesh & scale the grid
    break_the_cycle = 0
    start_iField = timing.time()
    meshfilename = '%s/%s.msh'%(ioctrl.path_test,ioctrl.sname)

    # Create the mesh and computational data
    M,CD,bc_int = mp._create_computational_mesh(meshfilename,ndim=2,mV=7,mP=3,mT=6)

    #scal the grid 
    M.x = M.x/sc.L
    M.y = M.y/sc.L
    M.xP = M.xP/sc.L
    M.yP = M.yP/sc.L
   
    # Create the pressure grid 
    M = compute_pressure_field_mesh(M,CD)
   
    #-
    # 1.b Create the boundary conditions
    
    BC = bc.bc_class(CD.NfemV,CD.NfemT,CD.NfemP,bc_int)
    
    # Create the phase data base
    
    M = generate_phase_field(M,BC)
    # - 
    # 1.c Create the solution class and precompute the shape functions
    
    S = sol.Sol(M,CD)
    S._precompute_shape_functions(CD)
    
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    print(f'->Creating the Mesh, BC and Sol took {timing.time()-start_iField:.3f} seconds')
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

    #-- Print the mesh information
    nq=CD.nq*M.nel
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    print ('NfemV  =', CD.NfemV)
    print ('NfemP  =', CD.NfemP)
    print ('Nfem   =', CD.NfemV+CD.NfemP)
    print ('nq     =', nq)
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

    # Checking if the shape function gives reasonable results: by checking if the area of the elements is correct, you check whether or not the shape
    # functions are correct. 
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

    #-- 2. Compute the area of the elements and compute the characteristic length of the elements 
 
    start = timing.time()
    
    S.velocities,S.scalars = sol.compute_area_elements(M,S.velocities,S.scalars,S.shapes,CD)
    
    print("compute elements areas: %.3f s" % (timing.time() - start))
    
    total_area_box = (np.max(M.x)-np.min(M.x))* (np.max(M.y)-np.min(M.y))
    
    error_area     = (total_area_box-np.sum(S.velocities.area))
    
    print(":::: area (m,M) %.6e %.6e [m]" %(np.min(S.velocities.area),np.max(S.velocities.area)))
    print(":::: total area %.6f [m^2]" %(np.sum(S.velocities.area)))
    print(":::: total area (Lx*Ly) %.6f [m^2]" %(total_area_box))
    print(":::: error is %.6f [m^2]"%(error_area))
    print(":::: relative error is %.3e []"%(error_area/total_area_box))
    
    if (error_area/total_area_box)*100>10:
        print('____________ relative error is high for velocity_________________')
    
    error_area     = (total_area_box-np.sum(S.scalars.area))
    
    print(":::: area T (m,M) %.6e %.6e [m]" %(np.min(S.scalars.area),np.max(S.scalars.area)))
    print(":::: total area %.6f [m^2]" %(np.sum(S.scalars.area)))
    print(":::: total area (Lx*Ly) %.6f [m^2]" %(total_area_box))
    print(":::: error is %.6f [m^2]"%(error_area))
    print(":::: relative error is %.3e []"%(error_area/total_area_box))
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

    # Set Boundary conditions, and initial condition for the temperature
    start = timing.time()
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    print("->Stokes BC took %.3f s" % (timing.time() - start))
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

    S = _generate_initial_setup_temperature(M,CD,BC,S,ctrl_n,lhs,sc)

    sh_v = shape_function_visual(CD)
    sh_v = _precompute_shape(sh_v,CD)


    triangulation = triangulation_visualisation(M,sc)

    time = 0.0 
    #-- Time loop/iteration loop 
    time_step = 0

    v_slab = 0.0 

    do_stokes = 0 

    # old_solution 


    res_u, res_v, res_T = 1.0, 1.0, 1.0

    it_main = 0 

    u_old = np.zeros_like(S.velocities.u)
    v_old = np.zeros_like(S.velocities.v)
    T_old = np.zeros_like(S.scalars.T)
    times = []
    timesteps_TD = []
    # initial guess solution: 
    _initial_guess_temperature(M,BC,S,CD,pdb,ctrl_n,lhs,bc_spec,sc)


    while time < ctrl_n.t_max:
        if ctrl_n.dt == 0: 
            print('Steady Solution')
            ctrl_n.t_max = 0.0 

        time_ts = timing.time()
        res_u, res_v, res_T = 1.0, 1.0, 1.0
        
        
        iterations_ss = []
        while ((res_u >= 1e-4) or (res_v >= 1e-4) or (res_T >= 1e-4)) :
            print(':::: _ MAIN NON LINEAR ITERATION _ ::::')
            print('=================================',it_main,'=================================')
            start_timing_nl = timing.time()
            if ctrl_n.dt > 0 :
                v_slab_pr = compute_slab(time,sc,ctrl_n)
                do_stokes = 0 
                if v_slab_pr != v_slab :
                    v_slab = compute_slab(time,sc,ctrl_n)
                    do_stokes = 1 
            if time_step == 0 or it_main == 0:
                BC = bc.set_boundary_condition_stokes(M,CD,BC,v_slab,bc_spec.Stokes,sc)

            #-- Compute the lithostatic pressure field 

            S.scalars.p_lit = sol.lithostatic_pressure(M,S,CD,BC,ctrl_n,pdb,it_main,sc)


            #-- Compute the Stokes system 

            if ctrl_n.rheology != 0 or time_step == 0 or do_stokes == 1:
                actvitate_non = 0 
                #if ctrl_n.time_dependent_v != 1:
                #    v_slab = np.linalg.norm(ctrl_n.v_s)

                BC = bc.set_boundary_condition_stokes(M,CD,BC,v_slab,bc_spec.Stokes,sc)

                if (it_main > 0) or (time_step > 0):
                    actvitate_non = 1 

                S,f = sol.solve_stokes_system(M,S,CD,BC,ctrl_n,pdb,actvitate_non,sc)

                do_stokes = 0 


            if ctrl_n.dt == 0.0:
            
                dt = ctrl_n.dt

            else:
            
                min_A = np.min(S.velocities.area)

                v_max = np.max(np.sqrt(S.velocities.u**2+S.velocities.v**2))

                kappa = 1e-6*(sc.T/sc.L**2)

                dt = np.min([np.sqrt(min_A)/(v_max),min_A/(2*kappa)])

                if time_step == 0: 
                    ctrl_n.dt = 0.3*dt
                else:
                    ctrl_n.dt = 0.7*dt


                print(".  -> dt = %.3f n.d., dt = %.2f Myr" % (ctrl_n.dt, ctrl_n.dt*sc.T/ctrl_n.scal_year/1e6))




            #-- Set the boundary conditions for the energy equation

            BC= bc.set_boundary_condition_energy(M,BC,S.velocities.u,S.velocities.v,ctrl_n,lhs,bc_spec.Energy,sc)


            #-- Solve the energy equation

            S = sol.solve_system_energy(M,S,CD,BC,ctrl_n,pdb,1,sc)

            # Check residuum: 
            if ctrl_n.dt!= 0.0:
                res_u = 1e-5
                res_v = 1e-5
                res_T = 1e-5 
            else: 
                res_u = np.linalg.norm(S.velocities.u-u_old,2)/np.linalg.norm(S.velocities.u+u_old) 
                res_v = np.linalg.norm(S.velocities.v-v_old,2)/np.linalg.norm(S.velocities.u+u_old)
                res_T = np.linalg.norm(S.scalars.T-T_old,2)/np.linalg.norm(S.scalars.T+T_old)


                relax = ctrl_n.relax
                print('. Res vel vx = %5e,vz = %5e'%(res_u,res_v))
                print('. Res vel T = %5e'%(res_T))
                print("    ---- ----    ")
                if it_main > 0: 
                    S.velocities.u = relax * S.velocities.u + (1-relax) * u_old
                    S.velocities.v = relax * S.velocities.v + (1-relax) * v_old
                    S.scalars.p = relax * S.scalars.p + (1-relax) * p_old
                    S.scalars.T = relax * S.scalars.T + (1-relax) * T_old
                    print("          MAX T, min T = [%.2f,%2f] degC"%(np.max(S.scalars.T*sc.Temp-273.15),np.min(S.scalars.T*sc.Temp-273.15)))
                    print("    ---- ----    ")
            
                    u_old = copy.deepcopy(S.velocities.u) 
                    v_old = copy.deepcopy(S.velocities.v)
                    T_old = copy.deepcopy(S.scalars.T)
                    p_old = copy.deepcopy(S.scalars.p)
                    end_timing_nl = timing.time()
                    print('=================================',it_main,'=================================')
                    print('time %.3f min'%((end_timing_nl-start_timing_nl)/60))
                    iterations_ss.append(it_main)
                    _output_routines(M,BC,S,CD,sh_v,pdb,ctrl_n,ioctrl,sc,triangulation,iterations_ss,time)

                it_main = it_main+1



        time = time + ctrl_n.dt
        time_step = time_step + 1 
        if time_step % 50 == 0 or time_step < 3 and ctrl_n.dt > 0.0 :
            print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
            print(f'. ->Time = {time*sc.T/ctrl_n.scal_year/1e6:.2f} Myr')
            print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
            times.append(time*sc.T/ctrl_n.scal_year/1e6)
            timesteps_TD.append(time_step)
            _output_routines(M,BC,S,CD,sh_v,pdb,ctrl_n,ioctrl,sc,triangulation,timesteps_TD,time*sc.T/ctrl_n.scal_year/1e6,times)


        time_B = timing.time()
        print(f'== --->{time_step} time step took {(time_ts-time_B)/60:.1f} min')

 

    End_iField = timing.time()
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    print('|||||---------->>>>>>iFieldStone finished its task in', (End_iField-start_iField)/60,' min')
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    return S,M,BC,CD 


