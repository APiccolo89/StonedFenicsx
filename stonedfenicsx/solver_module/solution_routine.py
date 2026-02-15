from stonedfenicsx.package_import import *

from stonedfenicsx.utils                     import timing_function, print_ph
from stonedfenicsx.material_property.compute_material_property import density_FX, heat_conductivity_FX, heat_capacity_FX, compute_viscosity_FX, compute_radiogenic 
from stonedfenicsx.create_mesh.aux_create_mesh   import Mesh,Domain
from stonedfenicsx.material_property.phase_db                  import PhaseDataBase
from stonedfenicsx.numerical_control         import NumericalControls, ctrl_LHS, IOControls
from stonedfenicsx.utils                     import interpolate_from_sub_to_main
from stonedfenicsx.scal  import Scal
from stonedfenicsx.output  import OUTPUT
from stonedfenicsx.utils  import compute_strain_rate
from stonedfenicsx.material_property.compute_material_property import Functions_material_properties_global, Functions_material_rheology
from stonedfenicsx.solver_module.problems_solution import Global_thermal, Global_pressure, Wedge, Slab, Solution
from stonedfenicsx.solver_module.solver_utilities import *

"""
Lame explanation: So, my idea is to conceive everything in a set of problems, without creating ad hoc classes for the wedge, subduction and so on
yes by the end I do, but at least I create problem class that solves stokes/scalar problems. Most of the surrounding stuff (controls/phase) are set, 
[Problem]=> uses Solver for solving a piece of the domain or the whole domain -> to increase the modularity I can introduce a class for the BC, but, 
to lazy to do it, but could be something that I can do after I have a full working code. 

=> Global Variables 
print yes|no 
-> import a python code with global variable is it possible? 

-> Apperently, following the Stokes tutorial, I was porting the generation of null space: it appears, that this trick, is not working together with null space
-> Nitsche Boundary condition constraints the pressure, and the null space dislike it, apperently because the simmetry of the formulation is wrong, but this is black
magic. 
ta  = fem.Function(P0, name="ta")        # target Function
expr = fem.Expression(rho, P0.element.interpolation_points())

ta.interpolate(expr)    


"""
def initialise_the_simulation(M:Mesh, 
                              ctrl:NumericalControls, 
                              lhs_ctrl:ctrl_LHS, 
                              pdb:PhaseDataBase, 
                              ioctrl:IOControls, 
                              sc:Scal)-> tuple[ctrl_LHS,
                                                Solution,
                                                Global_thermal,
                                                Global_pressure,
                                                Wedge,
                                                Slab,
                                                dolfinx.fem.function.Function,
                                                Functions_material_properties_global,
                                                Functions_material_rheology,
                                                Functions_material_rheology,
                                                Functions_material_rheology]:
    
    from stonedfenicsx.thermal_structure_ocean import compute_initial_LHS
    from stonedfenicsx.material_property.compute_material_property import populate_material_properties_thermal,populate_material_properties_rheology

    
    element_p           = M.element_p#basix.ufl.element("Lagrange","triangle", 1) 
    
    element_PT          = M.element_PT#basix.ufl.element("Lagrange","triangle",2)
    
    element_V           = M.element_V#basix.ufl.element("Lagrange","triangle",2,shape=(2,))

    #==================== Phase Parameter ====================
    lhs_ctrl = compute_initial_LHS(ctrl,lhs_ctrl, sc, pdb,M.g_input.theta_in_slab)  
          
    # Define Problem
    # Global energy
    energy_global = Global_thermal (M = M, name = ['energy','domainG']  , elements = (element_PT,), pdb = pdb, ctrl = ctrl)
    # Global lithostatic pressure
    lithostatic_pressure_global = Global_pressure(M = M, name = ['pressure','domainG'], elements = (element_PT,), pdb = pdb ) 
    # Wedge stokes problem
    wedge = Wedge(M =M, name = ['stokes','domainB'  ], elements = (element_V,element_p,element_PT), pdb = pdb)
    # Slab stokes problem
    slab = Slab(M = M, name = ['stokes','domainA'  ], elements = (element_V,element_p,element_PT))
    # Gravity, as I do not know where to put -> most likely inside the global problem 
    g = fem.Constant(M.domainG.mesh, PETSc.ScalarType([0.0, -ctrl.g]))    

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
    FGpdb   = populate_material_properties_thermal(FGpdb,pdb,M.domainG.phase)
    FGWG_R  = populate_material_properties_rheology(FGWG_R,pdb,M.domainB.phase)
    FGS_R   = populate_material_properties_rheology(FGS_R,pdb,M.domainA.phase)
    FGG_R   = populate_material_properties_rheology(FGG_R,pdb,M.domainG.phase)
    # Generate the initial guess for the temperature. 
    sol.T_O = energy_global.initial_temperature_field(M.domainG, ctrl, lhs_ctrl,M.g_input)
    

    
    return lhs_ctrl,sol,energy_global,lithostatic_pressure_global,slab,wedge,g,FGpdb,FGWG_R,FGS_R,FGG_R
#---------------------------------------------------------------------------------------------------
def outerloop_operation(M:Mesh,
                        ctrl:NumericalControls,
                        ctrlio:IOControls,
                        sc:Scal,
                        lhs:ctrl_LHS,
                        FGT:Functions_material_properties_global,
                        FGWR:Functions_material_rheology,
                        FGSR:Functions_material_rheology,
                        FGGR:Functions_material_rheology,
                        EG:Global_thermal,
                        LG:Global_pressure,
                        We:Wedge,
                        Sl:Slab,
                        sol:Solution,
                        g:dolfinx.fem.function.Function
                        ,ts:int=0)->Solution:
    
    # Initialise the it outer and residual outer 
    it_outer = 0 
    res      = 1
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
                                                    ,M.domainB.cell_par
                                                    ,1)
        
        sol.p_lwedge = interpolate_from_sub_to_main(sol.p_lwedge
                                                    ,sol.PL
                                                    ,M.domainB.cell_par
                                                    ,1)

        if it_outer == 0 and ts == 0: 
            sol = Sl.Solve_the_Problem(sol,
                                   ctrl
                                   ,FGSR
                                   ,M.domainA
                                   ,g
                                   ,sc,
                                   it = it_outer,
                                   ts=ts)

        if (We.typology == 'NonlinearProblem') or (We.typology == 'NonlinearProblemT') or (it_outer == 0):  
            sol = We.Solve_the_Problem(sol
                                ,ctrl
                                ,FGWR
                                ,M.domainB
                                ,g
                                ,sc
                                ,M.g_input
                                ,it = it_outer
                                ,ts=ts)

        # Interpolate from wedge/slab to global
        sol.u_global = interpolate_from_sub_to_main(sol.u_global
                                                    ,sol.u_wedge
                                                    , M.domainB.cell_par)
        sol.u_global = interpolate_from_sub_to_main(sol.u_global
                                                    ,sol.u_slab
                                                    , M.domainA.cell_par)
        
        
        sol.p_global = interpolate_from_sub_to_main(sol.p_global
                                                    ,sol.p_wedge
                                                    ,M.domainB.cell_par)
        
        sol.p_global = interpolate_from_sub_to_main(sol.p_global
                                                    ,sol.p_slab
                                                    ,M.domainA.cell_par)
        
        
        sol = EG.Solve_the_Problem(sol
                            ,ctrl
                            ,FGT
                            ,M
                            ,lhs
                            ,M.g_input
                            ,sc
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
                                     ,ts)


        print_ph('   // -- // :( --- ------- ------- ------- :) // -- // --- > ')

            
        it_outer = it_outer + 1
        
        
    
    return sol
#---------------------------------------------------------------------------------------------------
# Def time_loop 
def time_loop(M,ctrl,ioctrl,sc,lhs,FGT,FGWR,FGSR,FGGR,EG,LG,We,Sl,sol,g):
    
    if ctrl.steady_state == 1:
        print_ph('// -- // --- Steady   State  solution // -- // --- > ')
    else:
        print_ph('// -- // --- Time Dependent solution // -- // --- > ')

         
        
    t  = 0.0 
    ts = 0 
    output_class  = OUTPUT(M.domainG, ioctrl, ctrl, sc)
    
    # Initialise S.T_N 
    sol.T_N = sol.T_O.copy()
    
    while t<ctrl.time_max: 
        
        if ctrl.steady_state==0:
            print_ph(f'Time = {t*sc.T/sc.scale_Myr2sec:.3f} Myr, timestep = {ts:d}')
            print_ph('================ // =====================')
        
        # Prepare variable
        sol = outerloop_operation(M
                                  ,ctrl
                                  ,ioctrl
                                  ,sc
                                  ,lhs
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
                                  ,ts=ts)

        #if ctrl.adiabatic_heating==0:
        #    sol.T_ad = compute_adiabatic_initial_adiabatic_contribution(M.domainG,sol.T_N,None,sol.PL,FGT,0)

        if ctrl.steady_state == 1 or (ts%10) == 0:
            print_ph('OUTPUT...')
            output_class.print_output(sol,M.domainG,FGT,FGGR,ioctrl,sc,ctrl,it_outer=0,time=t*t*sc.T/sc.scale_Myr2sec,ts=ts)
            print_ph('finished')

        
    
        
        if ctrl.steady_state == 1: 
            print_ph('End Steady State solution, printing the benchmarks')
            t = ctrl.time_max
            if ctrl.van_keken == 1: 
                from stonedfenicsx.output import _benchmark_van_keken
                _benchmark_van_keken(sol,ioctrl,sc)

        if ctrl.steady_state == 0: 
            sol.t_oslab = interpolate_from_sub_to_main(sol.t_oslab
                                                    ,sol.T_O
                                                    ,M.domainA.cell_par
                                                    ,1)
            sol.t_nslab = interpolate_from_sub_to_main(sol.t_nslab
                                                    ,sol.T_O
                                                    ,M.domainA.cell_par
                                                    ,1)

        t = t+ctrl.dt
        
        sol.T_O = sol.T_N
        
        ts = ts + 1

    print_ph('Destroy Petsc Object and finish the simulation...')
    EG.solv.destroy()
    LG.solv.destroy()
    Sl.solv.destroy()
    We.solv.destroy()
    print_ph('---- The End ----')

    return 0 

#------------------------------------------------------------------------------------------------------------
def solution_routine(M:Mesh, ctrl:NumericalControls, lhs_ctrl:ctrl_LHS, pdb:PhaseDataBase, ioctrl:IOControls, sc:Scal):

    # Initialise
    (lhs_ctrl,                      # Left Boundary controls
    sol,                            # Solution data class
    EG,                  # Energy Problem defined in the global mesh
    LG,    # Lithostatic Problem defined in the global mesh
    Sl,                           # Stokes Problem defined in the slab mesh 
    We,                          # Stokes Problem defined in the wedge mesh
    g,                              # gravity 
    FGT,                          # Global thermal properties (pre-computed fem.function)
    FGWR,                         # Rheological material properties of the slab mesh
    FGSR,
    FGGR) = initialise_the_simulation(M,                 # Mesh 
                                       ctrl,              # Controls 
                                       lhs_ctrl,          # Not updated Lhs Control 
                                       pdb,               # Material property database
                                       ioctrl,            # Control input and output
                                       sc)                # Scaling 
    
    # Time Loop 
    
    time_loop(M,ctrl,ioctrl,sc,lhs_ctrl,FGT,FGWR,FGSR,FGGR,EG,LG,We,Sl,sol,g)
    
    return 0 
#--------------------------------------------------------------------------------------------
