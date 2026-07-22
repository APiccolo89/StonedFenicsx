

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
    """Instantiate all FEM problem objects and allocate the solution container.

    Constructs one problem object per physical sub-problem (global thermal,
    global lithostatic pressure, wedge Stokes, slab Stokes), creates their
    cached material-property tables, and allocates every fem.Function that
    will hold the evolving solution fields.  The initial temperature field is
    computed from the 1-D thermal boundary condition and stored in both
    sol.T_O (old) and sol.T_N (new) so the first Picard iteration starts
    from a physically meaningful guess.

    Args:
        ctrl_sim (SimulationControls): Container of all numerical, I/O, thermal
            and kinematic boundary-condition controls.
        pdb (PhaseDataBase): Material-property database for all phases.
        mesh (Mesh): Mesh object holding the global mesh, all sub-meshes,
            finite-element definitions, and geometric metadata.

    Returns:
        tuple[Solution, Global_thermal, Global_pressure, Wedge, Slab]:
            sol -- pre-allocated solution container with initial temperature.
            eg  -- global thermal energy problem.
            lg  -- global lithostatic pressure problem.
            sl  -- subducting-plate (slab) Stokes problem.
            we  -- mantle-wedge Stokes problem.
        object created: solution, energy, lithostatic pressure, slab, wedge
    """
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
                        ,ts:int=0)->None:
    """Execute one complete Picard outer-loop sweep over all coupled sub-problems.

    At each outer iteration the sub-problems are solved in the following order:
      1. Global lithostatic pressure  (lg) -- skipped after it_outer=0 if linear.
      2. Temperature and pressure interpolated from global to wedge/slab sub-meshes.
      3. Slab Stokes                  (sl) -- solved only at the first outer iteration
         of each timestep, or when the kinematic BC changes.
      4. Wedge Stokes                 (we) -- solved every iteration when nonlinear.
      5. Stokes solutions interpolated back to the global mesh.
      6. Global thermal energy        (eg) -- always solved.
      7. Outer-loop residual computed; loop exits when below ctrl_sim.ctrl.tol
         or ctrl_sim.ctrl.it_max iterations are reached.

    If all three sub-problems are linear the maximum number of outer iterations
    is capped at 2 (one solve + one residual check).

    Args:
        ctrl_sim (SimulationControls): Simulation controls (tolerances, max
            iterations, solver types, boundary-condition parameters).
        sc (Scal): Non-dimensionalisation scaling object.
        eg (Global_thermal): Global thermal energy problem.
        lg (Global_pressure): Global lithostatic pressure problem.
        we (Wedge): Mantle-wedge Stokes problem.
        sl (Slab): Subducting-plate Stokes problem.
        sol (Solution): Current solution container; updated in-place and returned.
        pdb (PhaseDataBase): Material-property database.
        ts (int, optional): Current timestep index. Defaults to 0.

    Returns:
        Solution: Updated solution container after the outer Picard loop has
        converged (or exhausted the iteration budget).
    """
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
            lg.Solve_the_Problem(sol,
                                       it_outer
                                       ,ts=ts)

        # Interpolate from global to wedge/slab

        interpolate_from_sub_to_main(sol.t_owedge
                                     ,sol.T_N
                                     ,we.domain.cell_par
                                     ,1)

        interpolate_from_sub_to_main(sol.p_lwedge
                                     ,sol.PL
                                     ,we.domain.cell_par
                                     ,1)

        if (ts == 0 and it_outer==0) or (it_outer == 0 and ctrl_sim.ctrl_ky.constant == 0): 
            sl.Solve_the_Problem(sol,
                                   it_outer = it_outer,
                                   ts=ts)

        if (we.typology == 'NonlinearProblem') or (we.typology == 'NonlinearProblemT') or (it_outer == 0):  
            we.Solve_the_Problem(sol=sol
                                ,it_outer = it_outer
                                ,ts=ts)

        # Interpolate from wedge/slab to global
        interpolate_from_sub_to_main(sol.u_global
                                      ,sol.u_wedge
                                      ,we.domain.cell_par)
        interpolate_from_sub_to_main(sol.u_global
                                      ,sol.u_slab
                                      , sl.domain.cell_par)
    
    
        interpolate_from_sub_to_main(sol.p_global
                                      ,sol.p_wedge
                                      ,we.domain.cell_par)
    
        interpolate_from_sub_to_main(sol.p_global
                                    ,sol.p_slab
                                    ,sl.domain.cell_par)
        
        eg.Solve_the_Problem(sol
                            ,it_outer = it_outer
                            ,ts = ts)
        
        # Compute residuum 
        res = compute_residuum_outer(sol=sol
                                     ,T=T_kouter
                                     ,PL=PL_kouter
                                     ,u=u_global_kouter
                                     ,p=p_global_kouter
                                     ,it_outer=it_outer
                                     ,sc=sc
                                     ,tA=time_A_outer
                                     ,ts=ts
                                     ,ctrl_sim=ctrl_sim
                                    )


        print_ph('|| --- || --- || --- || --- || --- || --- || --- || --- || --- || --- || --- || --- || ')

            
        it_outer = it_outer + 1
        
        
    
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
    """Drive the time-stepping loop and write output at the requested intervals.

    Instantiates the OUTPUT object once before entering the loop.  At every
    timestep:
      - Advances time-dependent boundary conditions (slab age, slab velocity)
        when ctrl_sim.ctrl_tbc.constant == 0 or ctrl_sim.ctrl_ky.constant == 0.
      - Calls outerloop_operation to converge the coupled Picard system.
      - Writes XDMF output every 10 timesteps (transient) or after the single
        steady-state solve.
      - Optionally runs the van Keken (2008) benchmark diagnostics.
      - Advances the physical time by ctrl_sim.ctrl.dt and rolls T_N → T_O.

    For steady-state mode (ctrl_sim.ctrl.steady_state == 1) the loop executes
    exactly once: the time variable is immediately set to time_max after the
    solve so the while condition is not re-entered.

    At the end of the loop all PETSc solver objects are explicitly destroyed to
    free GPU/CPU memory before the Python garbage collector runs.

    Args:
        ctrl_sim (SimulationControls): Simulation controls including time
            stepping (dt, time_max, steady_state), output frequency, and
            boundary-condition update flags.
        eg (Global_thermal): Global thermal energy problem (owns the XDMF
            domain and the cached thermal material table passed to OUTPUT).
        lg (Global_pressure): Global lithostatic pressure problem.
        we (Wedge): Mantle-wedge Stokes problem.
        sl (Slab): Subducting-plate Stokes problem.
        sol (Solution): Solution container carrying all field functions.
        pdb (PhaseDataBase): Material-property database (forwarded to
            outerloop_operation and OUTPUT).
        sc (Scal): Non-dimensionalisation scaling object.

    Returns:
        None
    """
    if ctrl_sim.ctrl.steady_state == 1:
        print_ph('|| --- || --- || Steady State solution || -- || --- || ')
    else:
        print_ph('|| --- || --- || Time Dependent solution || -- || --- || ')

         
        
    t  = 0.0 
    ts = 0 
    output_class  = OUTPUT(domain=eg.domain,ctrl_sim=ctrl_sim,sc=sc,pdb=pdb,cach_mat_thermal=eg.cached_mat,comm=eg.domain.mesh.comm)
    
    
    while t<ctrl_sim.ctrl.time_max: 
        time_A = timing.time()

        if ctrl_sim.ctrl.steady_state==0:
            print_ph(f'Time = {t*sc.time/sc.scale_myr2sec:.3f} Myr, timestep = {ts:d}')
            print_ph('||================ || =====================||')
            

        if ctrl_sim.ctrl_tbc.constant == 0: 
            ctrl_sim.ctrl_tbc.update_vel_age(t)
            ctrl_sim.ctrl_tbc.update_1d_vector_left()

            
        if ctrl_sim.ctrl_ky.constant == 0: 
            ctrl_sim.ctrl_ky.update_vel_age(t)


        # Prepare variable
        outerloop_operation(ctrl_sim=ctrl_sim
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
            output_class.print_output(sol=sol,ctrl_sim=ctrl_sim,sc=sc,ts=ts,it_outer=0,time=t*sc.time/sc.scale_myr2sec)
            print_ph('finished')

        
    
        
        if ctrl_sim.ctrl.steady_state == 1: 
            print_ph('End Steady State solution, printing the benchmarks')
            t = ctrl_sim.ctrl.time_max
            if eg.g_input.van_keken == 1: 
                from stonedfenicsx.output import _benchmark_van_keken
                _benchmark_van_keken(sol,ctrl_sim.ctrl_io,sc)

        if ts>0:
            t = t+ctrl_sim.ctrl.dt
            
    
        sol.T_O.x.array[:] = sol.T_N.x.array[:]
        sol.T_O.x.scatter_forward()
        
        time_B = timing.time()
        print_ph(f'              || -- ||Timestep {ts}  took {time_B-time_A:.2f} sec || -- || --- ||')

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
    """Top-level solver entry point: initialise and run the full simulation.

    Thin orchestration function that sequences the two main phases of the
    simulation:
      1. initialise_the_simulation -- allocates all FEM problem objects and
         the solution container.
      2. time_loop -- drives time stepping (or the single steady-state solve),
         Picard outer iteration, and output.

    This is the function called by stoned_fenicsx() after the configuration
    and mesh generation are complete.

    Args:
        ctrl_sim (SimulationControls): Container of all numerical, I/O,
            thermal-BC and kinematic-BC controls.  All parameters are already
            non-dimensionalised by the time this function is called.
        pdb (PhaseDataBase): Non-dimensionalised material-property database.
        mesh (Mesh): Mesh object holding the global and sub-domain meshes,
            element definitions, and geometric metadata.
        sc (Scal): Non-dimensionalisation scaling object used for output
            rescaling and time conversion.

    Returns:
        None
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
