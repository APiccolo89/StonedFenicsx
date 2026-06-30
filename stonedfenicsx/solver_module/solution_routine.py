

# --- 
from stonedfenicsx.config.numerical_control import SimulationControls
from stonedfenicsx.config.geometry import Mesh
from stonedfenicsx.config.scal import Scal
from stonedfenicsx.config.phase_db import PhaseDataBase
# --- 
from stonedfenicsx.utils import *
from stonedfenicsx.solver_module.problems_solution import Solution, Slab, Wedge, Global_thermal, Global_pressure
from stonedfenicsx.material_property.compute_material_property import populate_material_properties_thermal,populate_material_properties_rheology
# ---
import ufl 
import dolfinx 
import mpi4py as MPI
import petsc4py as petsc 
# --- 


def initialise_the_simulation(ctrl_sim:SimulationControls
                              ,pdb:PhaseDataBase
                              ,mesh:Mesh
                              ,sc:Scal)-> tuple[Global_pressure,Global_pressure,Wedge,Slab]:
    

    
    element_p           = mesh.element_p#basix.ufl.element("Lagrange","triangle", 1) 
    
    element_PT          = mesh.element_PT#basix.ufl.element("Lagrange","triangle",2)
    
    element_V           = mesh.element_V#basix.ufl.element("Lagrange","triangle",2,shape=(2,))
          
    # Define Problem
    # Global energy
    energy_global = Global_thermal (mesh = mesh, name = ['energy','domainG']  , elements = (element_PT,), pdb = pdb, ctrl_sim = ctrl_sim)
    # Global lithostatic pressure
    lithostatic_pressure_global = Global_pressure(mesh = mesh, name = ['pressure','domainG'], elements = (element_PT,), pdb = pdb ) 
    # Wedge stokes problem
    wedge = Wedge(M =M, name = ['stokes','domainB'  ], elements = (element_V,element_p,element_PT), pdb = pdb)
    # Slab stokes problem
    slab = Slab(M = M, name = ['stokes','domainA'  ], elements = (element_V,element_p,element_PT))
    # Gravity, as I do not know where to put -> most likely inside the global problem 
    g = femesh.Constant(mesh.domainG.mesh, PETSc.ScalarType([0.0, -ctrl.g]))    

    # Define Solution 
    # Create instance of solution.
    sol = Solution()
    # Allocate the function that handles. 
    sol.create_function(lithostatic_pressure_global,slab,wedge,[element_V,element_p])
    # Allocate the material properties.
    FGpdb   = Functions_material_properties_global()      
    FGWG_R  = Functions_material_rheology()
    FGS_R   = Functions_material_rheology()
    FGG_R   = Functions_material_rheology()
    # Populate the function.
    FGpdb   = populate_material_properties_thermal(FGpdb,pdb,mesh.domainG.phase)
    FGWG_R  = populate_material_properties_rheology(FGWG_R,pdb,mesh.domainB.phase)
    FGS_R   = populate_material_properties_rheology(FGS_R,pdb,mesh.domainA.phase)
    FGG_R   = populate_material_properties_rheology(FGG_R,pdb,mesh.domainG.phase)
    # Generate the initial guess for the temperature. 
    sol.T_O = energy_global.initial_temperature_field(mesh.domainG, ctrl, lhs_ctrl,mesh.g_input)
    

    
    return lhs_ctrl,sol,energy_global,lithostatic_pressure_global,slab,wedge,g,FGpdb,FGWG_R,FGS_R,FGG_R
#---------------------------------------------------------------------------------------------------
def outerloop_operation(M:Mesh,
                        ctrl:NumericalControls,
                        ctrlio:IOControls,
                        sc:Scal,
                        lhs:ctrl_LHS,
                        constant_vel:int
                        ,FGT:Functions_material_properties_global,
                        FGWR:Functions_material_rheology,
                        FGSR:Functions_material_rheology,
                        FGGR:Functions_material_rheology,
                        EG:Global_thermal,
                        LG:Global_pressure,
                        We:Wedge,
                        Sl:Slab,
                        sol:Solution
                        ,g:dolfinx.femesh.function.Function
                        ,pdb:PhaseDataBase
                        ,ts:int=0
                        ,ioctrl:IOControls=None)->Solution:
    
    # Initialise the it outer and residual outer 
    it_outer = 0 
    res      = 1
    debug = 0   
    if debug == 1: 
        out_deb = OUTPUT(mesh.domainG,ioctrl,ctrl,sc)
    
    if LG.typology == 'LinearProblem' and EG.typology == 'LinearProblem' and We.typology == 'LinearProblem':
        ctrl.it_max = 2
    
    
    while it_outer < ctrl.it_max and res > ctrl.tol: 
        
        print_ph(f'   // -- // --- Outer iteration {it_outer:d} for the coupled problem // -- // --- > ')
        
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
        
        
        if LG.typology == 'NonlinearProblem' or it_outer == 0:  
            sol = LG.Solve_the_Problem(sol,
                                       ctrl,
                                       FGT,
                                       M,
                                       g,
                                       it_outer,ts=ts)

        # Interpolate from global to wedge/slab

        sol.t_owedge = interpolate_from_sub_to_main(sol.t_owedge
                                                    ,sol.T_N
                                                    ,mesh.domainB.cell_par
                                                    ,1)
        
        sol.p_lwedge = interpolate_from_sub_to_main(sol.p_lwedge
                                                    ,sol.PL
                                                    ,mesh.domainB.cell_par
                                                    ,1)

        if (ts == 0 and it_outer==0) or (it_outer == 0 and constant_vel == 0): 
            sol = Sl.Solve_the_Problem(sol,
                                   ctrl
                                   ,FGSR
                                   ,mesh.domainA
                                   ,g
                                   ,sc,
                                   it = it_outer,
                                   ts=ts)

        if (We.typology == 'NonlinearProblem') or (We.typology == 'NonlinearProblemT') or (it_outer == 0):  
            sol = We.Solve_the_Problem(sol
                                ,ctrl
                                ,FGWR
                                ,mesh.domainB
                                ,g
                                ,sc
                                ,mesh.g_input
                                ,it = it_outer
                                ,ts=ts)

        # Interpolate from wedge/slab to global
        sol.u_global = interpolate_from_sub_to_main(sol.u_global
                                                    ,sol.u_wedge
                                                    , mesh.domainB.cell_par)
        sol.u_global = interpolate_from_sub_to_main(sol.u_global
                                                    ,sol.u_slab
                                                    , mesh.domainA.cell_par)
        
        
        sol.p_global = interpolate_from_sub_to_main(sol.p_global
                                                    ,sol.p_wedge
                                                    ,mesh.domainB.cell_par)
        
        sol.p_global = interpolate_from_sub_to_main(sol.p_global
                                                    ,sol.p_slab
                                                    ,mesh.domainA.cell_par)
        
        
        sol = EG.Solve_the_Problem(sol
                            ,ctrl
                            ,FGT
                            ,M
                            ,lhs
                            ,mesh.g_input
                            ,sc
                            ,pdb
                            ,it = it_outer
                            ,ts = ts)
        
        # Compute residuum 
        res,sol = compute_residuum_outer(sol
                                     ,T_kouter
                                     ,PL_kouter
                                     ,u_global_kouter
                                     ,p_global_kouter
                                     ,it_outer
                                     ,sc
                                     ,time_A_outer
                                     ,ctrl.Tmax
                                     ,ts
                                     ,ctrl)
        if debug==1:
            print_ph('OUTPUT...')
            out_deb.print_output(sol,mesh.domainG,FGT,FGGR,ioctrl,sc,ctrl,it_outer=it_outer,time=0,ts=ts,debug=1)
            print_ph('finished')


        print_ph('   // -- // :( --- ------- ------- ------- :) // -- // --- > ')

            
        it_outer = it_outer + 1
        
        
    
    return sol
#---------------------------------------------------------------------------------------------------
# Def time_loop 
def time_loop(M: Mesh
              ,ctrl: NumericalControls
              ,ioctrl:IOControls
              ,sc:Scal
              ,lhs:ctrl_LHS
              ,lhs_t:time_dependent_evolution
              ,FGT : Functions_material_properties_global
              ,FGWR : Functions_material_rheology
              ,FGSR : Functions_material_rheology
              ,FGGR : Functions_material_rheology
              ,EG : Global_thermal
              ,LG : Global_pressure
              ,We : Wedge 
              ,Sl : Slab
              ,sol : Solution
              ,g : dolfinx.femesh.Function
              ,pdb: PhaseDataBase
             ) -> None:
    """time loop function

    Args:
        M (Mesh): _description_
        ctrl (NumericalControls): _description_
        ioctrl (IOControls): _description_
        sc (Scal): _description_
        lhs (ctrl_LHS): _description_
        FGT (Functions_material_properties_global): _description_
        FGWR (Functions_material_rheology): _description_
        FGSR (Functions_material_rheology): _description_
        FGGR (Functions_material_rheology): _description_
        EG (Global_thermal): _description_
        LG (Global_pressure): _description_
        We (Wedge): _description_
        Sl (Slab): _description_
        sol (Solution): _description_
        g (dolfinx.femesh.Function): _description_
        t_ctrl (time_controls): _description_
    """
    if ctrl.steady_state == 1:
        print_ph('// -- // --- Steady   State  solution // -- // --- > ')
    else:
        print_ph('// -- // --- Time Dependent solution // -- // --- > ')

         
        
    t  = 0.0 
    ts = 0 
    output_class  = OUTPUT(mesh.domainG, ioctrl, ctrl, sc)
    
    # Initialise S.T_N 
    sol.T_N = sol.T_O.copy()
    
    while t<ctrl.time_max: 
        
        if ctrl.steady_state==0:
            print_ph(f'Time = {t*sc.T/sc.scale_Myr2sec:.3f} Myr, timestep = {ts:d}')
            print_ph('================ // =====================')
            

        if lhs_t.constant_age == 0: 
            from stonedfenicsx.thermal_structure_ocean import update_age_lhs
            
            lhs_t.current_age = lhs_t.update_vel_age(lhs_t.t_age,lhs_t.age_plate,t)
            lhs.c_age_plate = lhs_t.current_age
            lhs = update_age_lhs(ctrl
                                 ,lhs
                                 ,sc
                                 ,pdb
                                 ,mesh.g_input.theta_in_slab)
            print_ph(f'                            [{ts:d}]age plate = {lhs_t.current_age*sc.T/sc.scale_Myr2sec:3e} [Myr]')

            
        if lhs_t.constant_vel == 0: 
            lhs_t.current_vel = lhs_t.update_vel_age(lhs_t.t_vel,lhs_t.vel_plate,t)
            ctrl.v_s[0] = lhs_t.current_vel
            print_ph(f'                            [{ts:d}]velocity plate = {lhs_t.current_vel*(sc.L/sc.T)/sc.scale_vel:3e} [cm/yr]')
        
            
            
   
        # Prepare variable
        sol = outerloop_operation(M
                                  ,ctrl
                                  ,ioctrl
                                  ,sc
                                  ,lhs
                                  ,lhs_t.constant_vel
                                  ,FGT
                                  ,FGWR
                                  ,FGSR
                                  ,FGGR
                                  ,EG
                                  ,LG
                                  ,We
                                  ,Sl
                                  ,sol
                                  ,g
                                  ,pdb
                                  ,ts=ts
                                  ,ioctrl=ioctrl)



        if ctrl.steady_state == 1 or (ts%10) == 0:
            print_ph('OUTPUT...')
            output_class.print_output(sol,mesh.domainG,FGT,FGGR,ioctrl,sc,ctrl,it_outer=0,time=t*sc.T/sc.scale_Myr2sec,ts=ts)
            print_ph('finished')

        
    
        
        if ctrl.steady_state == 1: 
            print_ph('End Steady State solution, printing the benchmarks')
            t = ctrl.time_max
            if ctrl.van_keken == 1: 
                from stonedfenicsx.output import _benchmark_van_keken
                _benchmark_van_keken(sol,ioctrl,sc)

        t = t+ctrl.dt
            
    
        sol.T_O = sol.T_N
        
        ts = ts + 1

    print_ph('Destroy Petsc Object and finish the simulation...')
    EG.solv.destroy()
    LG.solv.destroy()
    Sl.solv.destroy()
    We.solv.destroy()
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
    initialise_the_simulation(ctrl_sim,pdb,mesh,sc)                # Scaling 
    
    # Time Loop 
    
    time_loop(M
              ,ctrl
              ,ioctrl
              ,sc
              ,lhs_ctrl
              ,lhs_t
              ,FGT
              ,FGWR
              ,FGSR
              ,FGGR
              ,EG
              ,LG
              ,We
              ,Sl
              ,sol
              ,g
              ,pdb)
    
    return 0 
#--------------------------------------------------------------------------------------------
