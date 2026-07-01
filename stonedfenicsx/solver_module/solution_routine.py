

# --- 
from stonedfenicsx.config.numerical_control import SimulationControls
from stonedfenicsx.config.geometry import Mesh
from stonedfenicsx.config.scal import Scal
from stonedfenicsx.config.phase_db import PhaseDataBase
# --- 
from stonedfenicsx.utils import interpolate_from_sub_to_main,timing,print_ph
from stonedfenicsx.solver_module.solver_utilities import compute_residuum_outer
from stonedfenicsx.solver_module.problems_solution import Solution, Slab, Wedge, Global_thermal, Global_pressure
from stonedfenicsx.output import OUTPUT

def initialise_the_simulation(ctrl_sim:SimulationControls = None
                              ,pdb:PhaseDataBase = None
                              ,mesh:Mesh = None
                              )-> tuple[Solution,Global_thermal,Global_pressure,Wedge,Slab]:    

    
    element_p           = mesh.element_p#basix.ufl.element("Lagrange","triangle", 1) 
    
    element_PT          = mesh.element_pt#basix.ufl.element("Lagrange","triangle",2)
    
    element_V           = mesh.element_v#basix.ufl.element("Lagrange","triangle",2,shape=(2,))
          
    # Define Problem
    # Global energy
    energy_global = Global_thermal (mesh = mesh, name = ['energy','global_domain']  , elements = (element_PT,), pdb = pdb, ctrl_sim = ctrl_sim)
    energy_global.create_cached_material(True)
    # Global lithostatic pressure
    lithostatic_pressure_global = Global_pressure(mesh = mesh, name = ['pressure','global_domain'], elements = (element_PT,), pdb = pdb, ctrl_sim=ctrl_sim ) 
    lithostatic_pressure_global.create_cached_material(True)
    # Wedge stokes problem
    wedge = Wedge(mesh =mesh, name = ['stokes','wedge_domain'  ], elements = (element_V,element_p,element_PT), pdb = pdb,ctrl_sim=ctrl_sim)
    wedge.create_cached_material(False)
    # Slab stokes problem
    slab = Slab(mesh = mesh, name = ['stokes','subduction_plate_domain'], elements = (element_V,element_p,element_PT),pdb=pdb,ctrl_sim=ctrl_sim)
    slab.create_cached_material(False)

    # Define Solution 
    # Create instance of solution.
    sol = Solution()
    # Allocate the function that handles. 
    sol.create_function(lithostatic_pressure_global,slab,wedge,[element_V,element_p])
    # Allocate the material properties.
    # Generate the initial guess for the temperature. 
    sol.T_O = energy_global.initial_temperature_field()
    sol.T_N = sol.T_O.copy()

    return sol, energy_global,lithostatic_pressure_global,slab,wedge
#---------------------------------------------------------------------------------------------------
def outerloop_operation(ctrl_sim:SimulationControls,
                        sc:Scal,
                        eg:Global_thermal,
                        lg:Global_pressure,
                        we:Wedge,
                        sl:Slab,
                        sol:Solution
                        ,pdb:PhaseDataBase
                        ,ts:int=0)->Solution:
    
    # Initialise the it outer and residual outer 
    it_outer = 0 
    res      = 1
    
    if lg.typology == 'LinearProblem' and eg.typology == 'LinearProblem' and we.typology == 'LinearProblem':
        ctrl_sim.ctrl.it_max = 2
    
    
    while it_outer < ctrl_sim.ctrl.it_max and res > ctrl_sim.ctrl.tol: 
        
        print_ph(f'   || -- || --- Outer iteration {it_outer:d} for the coupled problem || -- || --- || ')
        
        time_A_outer = timing.time()
        # Copy the old solution of the outer loop for computing the residual of the equations. 
        T_kouter        = sol.T_N.copy()
        T_kouter.x.scatter_forward()
        PL_kouter       = sol.PL.copy()
        PL_kouter.x.scatter_forward()
        u_global_kouter = sol.u_global.copy()
        u_global_kouter.x.scatter_forward()
        p_global_kouter = sol.p_global.copy()
        p_global_kouter.x.scatter_forward()
        
        
        if lg.typology == 'NonlinearProblem' or it_outer == 0:  
            sol = lg.Solve_the_Problem(sol,
                                       it_outer
                                       ,ts=ts)

        # Interpolate from global to wedge/slab

        sol.t_owedge = interpolate_from_sub_to_main(sol.t_owedge
                                                    ,sol.T_N
                                                    ,we.domain.cell_par
                                                    ,1)
        
        sol.p_lwedge = interpolate_from_sub_to_main(sol.p_lwedge
                                                    ,sol.PL
                                                    ,we.domain.cell_par
                                                    ,1)

        if (ts == 0 and it_outer==0) or (it_outer == 0 and ctrl_sim.ctrl_ky.constant == 0): 
            sol = sl.Solve_the_Problem(sol,
                                   it_outer = it_outer,
                                   ts=ts)

        if (we.typology == 'NonlinearProblem') or (we.typology == 'NonlinearProblemT') or (it_outer == 0):  
            sol = we.Solve_the_Problem(sol=sol
                                ,it = it_outer
                                ,ts=ts)

        # Interpolate from wedge/slab to global
        sol.u_global = interpolate_from_sub_to_main(sol.u_global
                                                    ,sol.u_wedge
                                                    ,we.domain.cell_par)
        sol.u_global = interpolate_from_sub_to_main(sol.u_global
                                                    ,sol.u_slab
                                                    , sl.domain.cell_par)
        
        
        sol.p_global = interpolate_from_sub_to_main(sol.p_global
                                                    ,sol.p_wedge
                                                    ,we.domain.cell_par)
        
        sol.p_global = interpolate_from_sub_to_main(sol.p_global
                                                    ,sol.p_slab
                                                    ,sl.domain.cell_par)
        
        
        sol = eg.Solve_the_Problem(sol
                            ,it = it_outer
                            ,ts = ts)
        
        # Compute residuum 
        res,sol = compute_residuum_outer(sol=sol
                                     ,T=T_kouter
                                     ,PL=PL_kouter
                                     ,u=u_global_kouter
                                     ,p=p_global_kouter
                                     ,it_outer=it_outer
                                     ,sc=sc
                                     ,time_A_outer=time_A_outer
                                     ,ts=ts
                                     ,ctrl_sim=ctrl_sim
                                    )


        print_ph('|| --- || --- || --- || --- || --- || --- || --- || --- || --- || --- || --- || --- || ')

            
        it_outer = it_outer + 1
        
        
    
    return sol
#---------------------------------------------------------------------------------------------------
# Def time_loop 
def time_loop(ctrl_sim:SimulationControls
              ,eg : Global_thermal
              ,lg : Global_pressure
              ,we : Wedge 
              ,sl : Slab
              ,sol : Solution
              ,pdb: PhaseDataBase
              ,sc: Scal
             ) -> None:

    if ctrl_sim.ctrl.steady_state == 1:
        print_ph('|| --- || --- || Steady State solution || -- || --- || ')
    else:
        print_ph('|| --- || --- || Time Dependent solution || -- || --- || ')

         
        
    t  = 0.0 
    ts = 0 
    output_class  = OUTPUT(domain=eg.domain,ctrl_sim=ctrl_sim,sc=sc,pdb=pdb,cach_mat_thermal=eg.cached_mat,comm=eg.domain.mesh.comm)
    
    
    while t<ctrl_sim.ctrl.time_max: 
        
        if ctrl_sim.ctrl.steady_state==0:
            print_ph(f'Time = {t*sc.temp/sc.scale_Myr2sec:.3f} Myr, timestep = {ts:d}')
            print_ph('||================ || =====================||')
            

        if ctrl_sim.ctrl_tbc.constant == 0: 
            ctrl_sim.ctrl_tbc.update_vel_age(t)
            ctrl_sim.ctrl_tbc.update_1d_vector_left()

            
        if ctrl_sim.ctrl_ky.constant == 0: 
            ctrl_sim.ctrl_ky.update_vel_age(t)

            
            
   
        # Prepare variable
        sol = outerloop_operation(ctrl_sim=ctrl_sim
                                  ,sc=sc
                                  ,eg=eg
                                  ,lg=lg
                                  ,we=we
                                  ,sl=sl
                                  ,sol=sol
                                  ,pdb=pdb
                                  ,ts=ts)



        if ctrl_sim.ctrl.steady_state == 1 or (ts%10) == 0:
            print_ph('OUTPUT...')
            output_class.print_output(sol=sol,ctrl_sim=ctrl_sim,sc=sc,ts=ts,it_outer=0,time=t*sc.temp/sc.scale_Myr2sec)
            print_ph('finished')

        
    
        
        if ctrl_sim.ctrl.steady_state == 1: 
            print_ph('End Steady State solution, printing the benchmarks')
            t = ctrl_sim.ctrl.time_max
            if ctrl_sim.ctrl.van_keken == 1: 
                from stonedfenicsx.output import _benchmark_van_keken
                _benchmark_van_keken(sol,ctrl_sim.ctrl_io,sc)

        t = t+ctrl_sim.ctrl.dt
            
    
        sol.T_O = sol.T_N
        
        ts = ts + 1

    print_ph('Destroy Petsc Object and finish the simulation...')
    eg.solv.destroy()
    lg.solv.destroy()
    sl.solv.destroy()
    we.solv.destroy()
    print_ph('---- The End ----')


#------------------------------------------------------------------------------------------------------------
def solution_routine(ctrl_sim:SimulationControls
                     ,pdb:PhaseDataBase
                     ,mesh:Mesh
                     ,sc:Scal
                    )->None:
    """Function that Initialise the object for the simulation (i.e., the problem, solution...)
    and send to the running simulations routine: outer_time_loop and inner_picard_loop. 

    Args:
        ctrl_sim (SimulationControls): Simulations controls [Numerical Controls, boundary controls, iocontrols]
        pdb (PhaseDataBase): Data Base of phase
        mesh (Mesh): Mesh and geometry object
        sc (Scal): Scaling object 

    """

    # Initialise
    sol,eg,lg,sl,we =initialise_the_simulation(ctrl_sim=ctrl_sim,pdb=pdb,mesh=mesh)                # Scaling 
    
    # Time Loop 
    
    time_loop(ctrl_sim=ctrl_sim
              ,eg = eg
              ,lg = lg
              ,we = we
              ,sl = sl 
              ,sol = sol 
              ,pdb=pdb
              ,sc=sc)
    
    return 0 
#--------------------------------------------------------------------------------------------
