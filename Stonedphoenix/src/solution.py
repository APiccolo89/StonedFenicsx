import numpy as np
import gmsh 

import ufl
import meshio
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx 
from dolfinx import mesh, fem, io, nls, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import matplotlib.pyplot as plt
from dolfinx.io import XDMFFile, gmshio
import gmsh 
from ufl import exp, conditional, eq, as_ufl
import scal as sc_f 
import basix.ufl
from utils import timing_function, print_ph
from compute_material_property import density_FX, heat_conductivity_FX, heat_capacity_FX, compute_viscosity_FX
from create_mesh import Mesh 
from phase_db import PhaseDataBase

#--------------------------------------------------------------------------------------------------------------
class Solution():
    def __init__(self):
        self.PL       : dolfinx.fem.function.Function 
        self.T_O      : dolfinx.fem.function.Function 
        self.T_N      : dolfinx.fem.function.Function 
        self.u_global : dolfinx.fem.function.Function
        self.u_wedge  : dolfinx.fem.function.Function
        self.p_lwedge : dolfinx.fem.function.Function
        self.t_owedge : dolfinx.fem.function.Function
        self.p_lslab  : dolfinx.fem.function.Function
        self.t_nslab  : dolfinx.fem.function.Function
        self.u_slab   : dolfinx.fem.function.Function
        self.p_global : dolfinx.fem.function.Function 
        self.p_wedge  : dolfinx.fem.function.Function 
        self.p_slab   : dolfinx.fem.function.Function
        
        
    def create_function(self,PG,PS,PW,elements): 
        """
        ======
        I am pretty pissed off about this problem: it is impossible to have a generalised approach that can be flexible 
        whoever had this diabolical idea, should be granted with a damnatio memoriae. So, inevitable, this class must 
        be modified as a function of the problem at hand.
        
        Argument: 
        PG : Global Problem 
        PS : Slab Problem
        PW : Wedge Problem
        
        -> update the solution functions that will be used later on. 
        
        =======
        """
        mixed_element = basix.ufl.mixed_element([elements[0], elements[1]])

        space_GL = fem.functionspace(PG.FS.mesh,mixed_element) # PORCO DIO 
        
        
        def gives_Function(space):
            Va = space.sub(0)
            Pa = space.sub(1)
            V,_  = Va.collapse()
            P,_  = Va.collapse()
            a = fem.Function(V)
            b = fem.Function(P)
            return a,b 
        
        self.PL       = fem.Function(PG.FS) # Thermal and Pressure problems share the same functional space -> Need to enforce this bullshit 
        self.T_O      = fem.Function(PG.FS) 
        self.T_N      = fem.Function(PG.FS)
        self.p_lwedge = fem.Function(PW.FSPT) # PW.SolPT -> It is the only part of this lovercraftian nightmare that needs to have temperature and pressure -> Viscosity depends pressure and temperature potentially
        self.t_nwedge = fem.Function(PW.FSPT) # same stuff as before, again, this is a nightmare: why the fuck. 
        self.u_global, self.p_global = gives_Function(space_GL)
        self.u_slab  , self.p_slab   = gives_Function(PS.FS)
        self.u_wedge , self.p_wedge  = gives_Function(PW.FS)

        return self 
#---------------------------------------------------------------------------
class Solvers():
    pass


class  ScalarSolver(Solvers):
    """
    class that store all the information for scalar like problems. Temperature and lithostatic pressure (potentially darcy like) are similar problem 
    they diffuse and advect a scalar. 
    --- 
    So -> Solver are more or less the same, I can store a few things and update as a function of the needs. 
    --- 
    Solve function require the problem P -> and a decision between linear and non linear -> form are handled by p class, so I do not give a fuck in this class 
    for now all the parameter will be default. 
    """
    def __init__(self,A ,b ,COMM, nl, J = None, r = None):
        self.A = fem.petsc.create_matrix(fem.form(A)) # Store the sparsisity 
        self.b = fem.petsc.create_vector(fem.form(b)) # Store the vector
        self.ksp = PETSc.KSP().create(COMM)           # Create the ksp object 
        self.ksp.setOperators(self.A)                # Set Operator
        self.ksp.setType("gmres")
        self.ksp.setTolerances(rtol=1e-4, atol=1e-3)
        self.pc = self.ksp.getPC()
        self.pc.setType("lu")
        self.pc.setFactorSolverType("mumps")
        if nl == 1: 
            self.J = fem.petsc.create_matrix(fem.form(J))
            self.r = fem.petsc.create_vector(fem.form(r))
            
        else: 
            self.J = None
            self.r = None 
    
    


#class SolverStokes(): 



        
        
        
        
        
        

    
#----------------------------------------------------------------------------     
class Problem:
    name      : list                               # name of the problem, domain [global, domainA...]
    mixed     : bool                               # is a mixed problem (e.g. Stokes problem has two function spaces: velocity and pressure)
    FS        : dolfinx.fem.FunctionSpace          # Function space of the problem 
    F0        : dolfinx.fem.FunctionSpace | None   # Function space of the subspace 
    F1        : dolfinx.fem.FunctionSpace | None   # Function space of the subspace
    trial0    : ufl.Argument | None                # Trial 
    trial1    : ufl.Argument | None                # Trial
    test0     : ufl.Argument | None                # Test
    test1     : ufl.Argument | None                # Test 
    typology  : str | None                         # Linear/Non Linear
    dofs      : np.ndarray | None                  # Boundary dofs
    bc        : list                               # Dirichlet BC list
    ds        : ufl.measure.Measure                # measure surface/length 
    solv      : Solvers
    # --
    def __init__(self, M: Mesh, elements: tuple, name: list):
        """
        Arguments: 
            M       : the mesh object 
            elements: tuple containing the elements.
                      If >1 -> assumes mixed problem
            name    : list ['problem_name', 'domain']
        """
        self.name = name 
        if name[1] not in ("domainA", "domainB", "domainC", "domainG"):
            raise NameError("Wrong domain name, check the spelling, in my case was it")
        elif name[1] == "domainC":
            print("Are you sure? DomainC is junk for this problem.")

        M = getattr(M, name[1])

        if len(elements) == 1: 
            self.mixed    = False
            self.FS       = dolfinx.fem.functionspace(M.mesh, elements[0]) 
            self.trial0   = ufl.TrialFunction(self.FS)
            self.test0    = ufl.TestFunction(self.FS)
            self.trial1   = None
            self.test1    = None
        else: 
            self.mixed    = True
            mixed_element = basix.ufl.mixed_element([elements[0], elements[1]])
            self.FS       = dolfinx.fem.functionspace(M.mesh, mixed_element) # MA cristoiddio, perche' cazzo hanno messo FunctionSpace and functionspace come nomi, ma sono degli stronzi?
            # 
            self.F0       = self.FS.sub(0)
            self.F1       = self.FS.sub(1)
            # Define trial/test on mixed FS
            trial         = ufl.TrialFunction(self.FS)
            test          = ufl.TestFunction(self.FS)
            self.trial0   = trial[0]
            self.trial1   = trial[1]
            self.test0    = test[0]
            self.test1    = test[1]    
         
        

        
#------------------------------------------------------------------
class Global_thermal(Problem):
    def __init__(self,M:Mesh, elements:tuple, name:list,pdb:PhaseDataBase):
        super().__init__(M,elements,name)
#-----------------------------------------------------------------
class Global_pressure(Problem): 
    def __init__(self,M:Mesh, elements:tuple, name:list,pdb:PhaseDataBase):
        super().__init__(M,elements,name)
        if np.all(pdb.option_rho<2):
            self.typology = 'LinearProblem'
        else:
            self.typology = 'NonlinearProblem'

        self.bc = [self.set_problem_bc(getattr(M,'domainG'))]
    def set_problem_bc(self,M):
         
        fdim = 1    
        top_facets   = M.facets.find(M.bc_dict['Top'])
        top_dofs    = fem.locate_dofs_topological(self.FS, 1, top_facets)
        bc = [fem.dirichletbc(0.0, top_dofs, self.FS)]
        return bc  
    
    def set_newton(self,p,D,T,g):
        test   = self.test0                 # v
        trial0 = self.trial0                # δp (TrialFunction) for Jacobian

        a_lin = ufl.inner(ufl.grad(p), ufl.grad(test)) * ufl.dx
        L     = ufl.inner(ufl.grad(test),
                              density_FX(pdb, T, p, D.phase, Dmesh) * g) * ufl.dx
        # Nonlinear residual: F(p; v) = ∫ ∇p·∇v dx - ∫ ∇v·(ρ(T, p) g) dx
        F = a_lin - L
        # Jacobian dF/dp in direction δp (trial0)
        J = ufl.derivative(F, p, trial0)
        
        return F,J 
    
    
    def set_linear_picard(self,p_k,T,D,pdb,g):
        # Function that set linear form and linear picard for picard iteration
        
        rho_k = density_FX(pdb, T, p_k, D.phase, D.mesh)  # frozen
        
        # Linear operator with frozen coefficients
        
        a = ufl.inner(ufl.grad(self.trial0), ufl.grad(self.test0)) * ufl.dx
        
        L = ufl.inner(ufl.grad(self.test0), rho_k * g) * ufl.dx
        

        return a, L

    def Solve_the_Problem(self,S,ctrl,pdb,M,g,it=0,ts=0): 
        
        nl = 0 
        p_k = S.PL.copy()  # Previous lithostatic pressure 
        T   = S.T_O # -> will not eventually update 
        
        # If the problem is linear, p_k is not doing anything, it is there because I
        # design the density function to receive in anycase a pressure, potentially I
        # can use the multipledispach of python, which say ah density with pressure 
        # density without pressure is equal to fuck. But seems a bit lame, and I do 
        # not think that is a great improvement of the code. 
        
        a,L = self.set_linear_picard(p_k,T,getattr(M,'domainG'),pdb,g)
        
        if self.typology == 'NonlinearProblem':
            F,J = self.set_newton(p_k,D,T,p,g)
            nl = 1 
        else: 
            F = None;J=None 

        if it == 0 & ts == 0: 
            self.solv = ScalarSolver(a,L,M.comm,nl,J,F)
        
        if nl == 0: 
            S = self.solve_the_linear(S,a,L) 
        else: 
            S = self.solve_the_non_linear(S,ctrl,pdb)

        return S 
    
    def solve_the_linear(self,S,a,L,isPicard=0,it=0,ts=0):
        
        buf = fem.Function(self.FS)
        x   = self.solv.b.copy()
        if isPicard == 0 or it == 0 or ts == 0:
            self.solv.A.zeroEntries()
            fem.petsc.assemble_matrix(self.solv.A,fem.form(a),self.bc[0])
            self.solv.A.assemble()
        # b -> can change as it is the part that depends on the pressure in case of nonlinearities
        self.solv.b.set(0.0)
        fem.petsc.assemble_vector(self.solv.b, fem.form(L))
        fem.petsc.apply_lifting(self.solv.b, [fem.form(a)], self.bc)
        self.solv.b.ghostUpdate()
        fem.petsc.set_bc(self.solv.b, self.bc[0])
        self.solv.ksp.solve(self.solv.b, x)
        x.x.scatter_forward()
        
        if isPicard == 0: # if it is a picard iteration the function gives the function 
            S.PL = x.copy(deepcopy==True)
            return S 
        else:
            return x 
    
    def solve_the_non_linear():  
        pass 
        
        

                
        
        
        
        
        
#-----------------------------------------------------------------
class Wedge(Problem): 
    def __init__(self,M:Mesh, elements:tuple, name:list,pdb:PhaseDataBase):
        super().__init__(M,elements,name)
        M = getattr(M,name[1])
        self.FSPT = dolfinx.fem.functionspace(M.mesh, elements[2])  # Needed for accounting the pressure. 
#------------------------------------------------------------------
class Slab(Problem): 
    def __init__(self,M:Mesh, elements:tuple, name:list,pdb:PhaseDataBase):
        super().__init__(M,elements,name)
#-------------------------------------------------------------------
@timing_function
def solve_lithostatic_problem(S, pdb, sc, g, M ):
    """
    PL  : function
    T_o : previous Temperature field

    pdb : phase data base 
    sc  : scaling 
    g   : gravity vector 
    M   : Mesh object 
    --- 
    Output: current lithostatic pressure. 
    
    To do: Improve the solver options and make it faster
    create an utils function for timing. 
    
    """
    print_ph("[] - - - -> Solving Lithostatic pressure problem <- - - - []")

    tPL = ufl.TrialFunction(M.Sol_SpaceT)  
    TPL = ufl.TestFunction(M.Sol_SpaceT)
    
    flag = 1 
    fdim = M.mesh.topology.dim - 1    
    top_facets   = M.mesh_Ftag.find(1)
    top_dofs    = fem.locate_dofs_topological(M.Sol_SpaceT, M.mesh.topology.dim-1, top_facets)
    bc = fem.dirichletbc(0.0, top_dofs, M.Sol_SpaceT)
    
    # -> yeah only rho counts here. 
    if (np.all(pdb.option_rho) == 0):
        flag = 0
        bilinear = ufl.dot(ufl.grad(TPL), ufl.grad(tPL)) * ufl.dx
        linear   = ufl.dot(ufl.grad(TPL), density_FX(pdb, S.T_O, S.PL, M.phase,M.mesh)*g) * ufl.dx
        problem = fem.petsc.LinearProblem(bilinear, linear, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu","pc_factor_mat_solver_type": "mumps"})
        
        S.PL = problem.solve()
                
    
    if flag !=0: 
        # Solve lithostatic pressure - Non linear 
        F = ufl.dot(ufl.grad(S.PL), ufl.grad(TPL)) * ufl.dx - ufl.dot(ufl.grad(TPL), density_FX(pdb, S.T_O, S.PL, M.phase,M.mesh)*g) * ufl.dx

        problem = fem.petsc.NonlinearProblem(F, S.PL, bcs=[bc])
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-4
        solver.report = True
        ksp = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "gmres"
        opts[f"{option_prefix}ksp_rtol"] = 1.0e-4
        ksp.setFromOptions()
        n, converged = solver.solve(PL)
    local_max = np.max(S.PL.x.array)

    # Global min and max using MPI reduction
    global_max = M.comm.allreduce(local_max, op=MPI.MAX)
    print_ph(f"// - - - /Global max lithostatic pressure is    : {global_max*sc.stress/1e9:.2f}[GPa]/")
    print_ph("               ")
    print_ph("               _")
    print_ph("               :")
    print_ph("[] - - - -> Finished <- - - - []")
    print_ph("               :")
    print_ph("               _")
    print_ph("                ")

    from utils import interpolate_from_sub_to_main
    
    S.p_lwedge = interpolate_from_sub_to_main(S.p_lwedge,S.PL,M.domainB.scell,1)
    S.p_lslab  = interpolate_from_sub_to_main(S.p_lslab,S.PL,M.domainA.scell,1)

    return S  


def set_ups_problem():
    from phase_db import PhaseDataBase
    from phase_db import _generate_phase
    from thermal_structure_ocean import compute_initial_LHS
    from scal import Scal 
    import scal as sc_f 
    from numerical_control import IOControls,NumericalControls,ctrl_LHS
    
    from create_mesh import unit_test_mesh
    # Create scal 
    sc = Scal(L=660e3,Temp = 1350,eta = 1e21, stress = 1e9)
    
    ioctrl = IOControls(test_name = 'Debug_test',
                        path_save = '../Debug_mesh',
                        sname = 'Experimental')
    ioctrl.generate_io()
    # Create mesh 
    M = unit_test_mesh(ioctrl, sc)
            
    print_ph("[] - - - -> Creating numerical controls <- - - - []")
    ctrl = NumericalControls()
    
    ctrl = sc_f._scaling_control_parameters(ctrl, sc)
    
    print_ph("[] - - - -> Phase Database <- - - - []")    
        
    # This function can be a canvas for the main solver update. 
    # Define elements -> would be defined in the controls 

    element_p           = basix.ufl.element("Lagrange","triangle", 1) 
    
    element_PT          = basix.ufl.element("Lagrange","triangle",2)
    
    element_V           = basix.ufl.element("Lagrange","triangle",2,shape=(2,))
    
    #===============    
    # Define PhaseDataBase
    pdb = PhaseDataBase(6)
    # Slab
    
    pdb = _generate_phase(pdb, 1, rho0 = 3300 , option_rho = 0, option_rheology = 0, option_k = 0, option_Cp = 0, eta=1e22)
    # Oceanic Crust
    
    pdb = _generate_phase(pdb, 2, rho0 = 2900 , option_rho = 0, option_rheology = 0, option_k = 0, option_Cp = 0, eta=1e21)
    # Wedge
    
    pdb = _generate_phase(pdb, 3, rho0 = 3300 , option_rho = 0, option_rheology = 3, option_k = 0, option_Cp = 0, eta=1e21)#, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    # 
    
    pdb = _generate_phase(pdb, 4, rho0 = 3250 , option_rho = 0, option_rheology = 0, option_k = 0, option_Cp = 0, eta=1e23)#, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    #
    
    pdb = _generate_phase(pdb, 5, rho0 = 2800 , option_rho = 0, option_rheology = 0, option_k = 0, option_Cp = 0, eta=1e21)#, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    #
    
    pdb = _generate_phase(pdb, 6, rho0 = 2700 , option_rho = 0, option_rheology = 0, option_k = 0, option_Cp = 0, eta=1e21)#, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    #
    
    pdb = sc_f._scaling_material_properties(pdb,sc)
    # Define LHS 

    lhs_ctrl = ctrl_LHS()

    lhs_ctrl = sc_f._scale_parameters(lhs_ctrl, sc)
    
    lhs_ctrl = compute_initial_LHS(ctrl,lhs_ctrl, sc, pdb)  
          
    # Define Problem
    
    # Pressure 
    
    lithostatic_pressure_global = Global_pressure(M = M, name = ['pressure','domainG'], elements = (element_PT,                   ), pdb = pdb)
    
    wedge                       = Wedge          (M = M, name = ['stokes','domainB'  ], elements = (element_V,element_p,element_PT), pdb = pdb)
    
    slab                        = Slab           (M = M, name = ['stokes','domainA'  ], elements = (element_V,element_p           ), pdb = pdb)
    
    g = fem.Constant(M.domainG.mesh, PETSc.ScalarType([0.0, -ctrl.g]))    

    # Define Solution 
    sol                         = Solution()
    
    sol.create_function(lithostatic_pressure_global,slab,wedge,[element_V,element_p])
    
    lithostatic_pressure_global.Solve_the_Problem(sol,ctrl,pdb,M,g,it=0,ts=0)
    
    
    
    
    # Update and set up variational problem
    
    # non-linear itearion betwee pressure/stk/temperature 
    #===============
    # -> [a] Solve Pressure: newton picard+newton
    #        [/] interpolate solutions from global to local 
    #    [b] Solve Slab if (it == 0 or time_step == 0) & vel_time == 1 (mix Picard & Newton)=> iterate picard till 1e-3 -> try newton  
    #    [c] Solve Wedge if (it==0  or time_step == 0) & (rheo == nonlinear or vel_time == 1)
    #        [/] interpolate solutions from local to global 
    #    [d] Solve thermal problem -> steady state / time dependent
    #    [e] Check residuum of PL,T,u(G),p(G) -> if variation of norm(dT,dPL,du(G),dp(G))<tol -> iteration finished 
    # ==============         
    #    [Output] : interpoalte solution variable DG1 -> save output -> Extract additional data 
    #             : density,k,Cp,viscosity, strain rate, pressure 
    #    [->done<-] Create relational database, {Here I need to do a decent job, for showing off my skills to potential hiring, like the manager of an obnoxious McDonald}
    # -------------
    
    
    

    
    return 0 



if __name__ == '__main__': 
    
    set_ups_problem()