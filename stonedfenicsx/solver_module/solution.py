from .package_import import *

from .utils                     import timing_function, print_ph
from .compute_material_property import density_FX, heat_conductivity_FX, heat_capacity_FX, compute_viscosity_FX, compute_radiogenic 
from .create_mesh.aux_create_mesh   import Mesh,Domain
from .phase_db                  import PhaseDataBase
from dolfinx.fem.petsc          import assemble_matrix_block, assemble_vector_block
from .numerical_control         import NumericalControls, ctrl_LHS, IOControls
from .utils                     import interpolate_from_sub_to_main
from .scal                      import Scal
from .output                    import OUTPUT
from .utils                     import compute_strain_rate
from .compute_material_property import Functions_material_properties_global, Functions_material_rheology


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
#--------------------------------------------------------------------------------------------------------------
class Solution():
    def __init__(self):
        self.PL       : dolfinx.fem.function.Function 
        self.T_i      : dolfinx.fem.function.Function
        self.T_O      : dolfinx.fem.function.Function 
        self.T_N      : dolfinx.fem.function.Function 
        self.u_global : dolfinx.fem.function.Function
        self.u_wedge  : dolfinx.fem.function.Function
        self.p_lwedge : dolfinx.fem.function.Function
        self.t_owedge : dolfinx.fem.function.Function
        self.p_lslab  : dolfinx.fem.function.Function
        self.t_oslab  : dolfinx.fem.function.Function
        self.u_slab   : dolfinx.fem.function.Function
        self.p_global : dolfinx.fem.function.Function 
        self.p_wedge  : dolfinx.fem.function.Function 
        self.p_slab   : dolfinx.fem.function.Function
        self.Hs_wedge : dolfinx.fem.function.Function
        self.Hs_slab  : dolfinx.fem.function.Function
        self.Hs_global : dolfinx.fem.function.Function
        self.T_ad      : dolfinx.fem.function.Function
        self.outer_iteration : NDArray[:]
        self.mT            : NDArray[:]
        self.MT            : NDArray[:]     
        self.ts            : NDArray[:]   
        
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
            P,_  = Pa.collapse()
            a = fem.Function(V)
            b = fem.Function(P)
            return a,b 
        
        self.PL       = fem.Function(PG.FS) # Thermal and Pressure problems share the same functional space -> Need to enforce this bullshit 
        self.T_O      = fem.Function(PG.FS) 
        self.T_N      = fem.Function(PG.FS)
        self.Hs_global = fem.Function(PG.FS)
        self.Hs_slab  = fem.Function(PS.FSPT)
        self.Hs_wedge = fem.Function(PW.FSPT)
        self.T_i      = fem.Function(PG.FS)
        self.p_lwedge = fem.Function(PW.FSPT) # PW.SolPT -> It is the only part of this lovercraftian nightmare that needs to have temperature and pressure -> Viscosity depends pressure and temperature potentially
        self.t_owedge = fem.Function(PW.FSPT) # same stuff as before, again, this is a nightmare: why the fuck. 
        self.p_lslab = fem.Function(PS.FSPT) # PW.SolPT -> It is the only part of this lovercraftian nightmare that needs to have temperature and pressure -> Viscosity depends pressure and temperature potentially
        self.t_oslab = fem.Function(PS.FSPT)
        self.t_nslab = fem.Function(PS.FSPT)
        self.u_global, self.p_global = gives_Function(space_GL)
        self.u_slab  , self.p_slab   = gives_Function(PS.FS)
        self.u_wedge , self.p_wedge  = gives_Function(PW.FS)
        self.T_ad                     = fem.Function(PG.FS)   
        self.mT    = np.zeros(1,dtype=float)
        self.MT    = np.zeros(1,dtype=float) 
        self.outer_iteration = np.zeros(1,dtype=float)
        self.ts             = np.zeros(1,dtype=int)

        return self 
#---------------------------------------------------------------------------
        
#-------------------------------------------------------------------------       
def decoupling_function(z,fun,g_input):
    """
    Function explanation: 
    [a]: z depth coordinate 
    [b]: fun a function 
    [c]: ctrl => deprecated 
    [d]: D => deprecated 
    [e]: g_input -> still here, geometric parameters
    
    
    
    """
    
    dc = g_input.decoupling
    lit = g_input.ns_depth
    dc = dc/g_input.decoupling
    lit = lit/g_input.decoupling
    z2 = np.abs(z)/g_input.decoupling
    trans = g_input.trans/g_input.decoupling
    

    fun.x.array[:] = 1-0.5 * ((1)+(1)*np.tanh((z2-dc)/(trans/4)))
    
    
    return fun
        


 
        

        

class Global_pressure(Problem): 
    def __init__(self,M:Mesh, elements:tuple, name:list,pdb:PhaseDataBase):
        super().__init__(M,elements,name)
        if np.all(pdb.option_rho<2):
            self.typology = 'LinearProblem'
        else:
            self.typology = 'NonlinearProblem'

        self.bc = [self.set_problem_bc(getattr(M,'domainG'))]
    def set_problem_bc(self,M):
         
        top_facets   = M.facets.find(M.bc_dict['Top'])
        top_dofs    = fem.locate_dofs_topological(self.FS, 1, top_facets)
        bc = [fem.dirichletbc(0.0, top_dofs, self.FS)]
        return bc  
    
    def set_newton(self,p,D,T,g,pdb):
        test   = self.test0                 # v
        trial0 = self.trial0                # δp (TrialFunction) for Jacobian

        a_lin = ufl.inner(ufl.grad(p), ufl.grad(test)) * self.dx
        L     = ufl.inner(ufl.grad(test),
                              density_FX(pdb, T, p) * g) * self.dx
        # Nonlinear residual: F(p; v) = ∫ ∇p·∇v dx - ∫ ∇v·(ρ(T, p) g) dx
        F = a_lin - L
        # Jacobian dF/dp in direction δp (trial0)
        J = ufl.derivative(F, p, trial0)
        
        return F,J 
    
    
    def set_linear_picard(self,p_k,T,D,FG,g, it=0):
        # Function that set linear form and linear picard for picard iteration
        
        rho_k = density_FX(FG, T, p_k)  # frozen
        
        # Linear operator with frozen coefficients
        if it == 0: 
            a = ufl.inner(ufl.grad(self.trial0), ufl.grad(self.test0)) * self.dx
        else: 
            a = None
        
        L = ufl.inner(ufl.grad(self.test0), rho_k * g) * self.dx
        

        return a, L

    def Solve_the_Problem(self,S,ctrl,pdb,M,g,it_outer=0,ts=0): 
        
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
            F,J = self.set_newton(p_k,getattr(M,'domainG'),T,g,pdb)
            nl = 1 
        else: 
            F = None;J=None 

        if it_outer == 0 & ts == 0: 
            self.solv = ScalarSolver(a,L,M.comm,nl,J,F)
        
        print_ph('              // -- // --- LITHOSTATIC PROBLEM [GLOBAL] // -- // --- > ')

        time_A = timing.time()

        
        if nl == 0: 
            S = self.solve_the_linear(S,a,L,S.PL) 
        else: 
            S = self.solve_the_non_linear(M,S,ctrl,pdb,g)

        time_B = timing.time()

        print_ph('              // -- // --- Solution of Lithostatic pressure problem finished in {time_B-time_A:.2f} sec // -- // --->')
        print_ph('')

        return S 
    
    def solve_the_linear(self,S,a,L,function_fen,isPicard=0,it=0,ts=0):
        
        if it == 0 or ts == 0:
            self.solv.A.zeroEntries()
            fem.petsc.assemble_matrix(self.solv.A,fem.form(a),self.bc[0])
            self.solv.A.assemble()
        # b -> can change as it is the part that depends on the pressure in case of nonlinearities
        self.solv.b.set(0.0)
        fem.petsc.assemble_vector(self.solv.b, fem.form(L))
        fem.petsc.apply_lifting(self.solv.b, [fem.form(a)], self.bc)
        self.solv.b.ghostUpdate()
        fem.petsc.set_bc(self.solv.b, self.bc[0])
        self.solv.ksp.solve(self.solv.b, function_fen.x.petsc_vec)
        function_fen.x.scatter_forward()
        
        
        if isPicard == 0: # if it is a picard iteration the function gives the function 
            S.PL = function_fen 
            return S
        else:
            return function_fen
    
    def solve_the_non_linear(self,M,S,ctrl,FG,g):  
        
        tol = 1e-3  # Tolerance Picard  
        tol = 1.0 
        p_k = S.PL.copy() 
        p_k1 = S.PL.copy()
        du   = S.PL.copy()
        du2  = S.PL.copy()
        it_inner = 0 
        
        
        print_ph('              [//] Picard iterations for the non linear lithostatic pressure problem')

        
        while it_inner < ctrl.it_max and tol > ctrl.tol_innerPic:
            time_ita = timing.time()
            
            if it_inner == 0:
                A,L = self.set_linear_picard(p_k,S.T_O,getattr(M,'domainG'),FG,g)
            else: 
                _,L = self.set_linear_picard(p_k,S.T_O,getattr(M,'domainG'),FG,g,1)
            
            p_k1 = self.solve_the_linear(S,A,L,p_k1,1,it_inner,1) 

            # L2 norm 
            du.x.array[:]  = p_k1.x.array[:] - p_k.x.array[:];du.x.scatter_forward()
            du2.x.array[:] = p_k1.x.array[:] + p_k.x.array[:];du2.x.scatter_forward()
            tol= L2_norm_calculation(du)/L2_norm_calculation(du2)
            
            time_itb = timing.time()
            print_ph(f'              []L_2 norm is {tol:.3e}, it_th {it_inner:d} performed in {time_itb-time_ita:.2f} seconds')
            
            #update solution
            p_k.x.array[:] = p_k1.x.array[:]*0.7 + p_k.x.array[:]*(1-0.7)
            
            it_inner = it_inner + 1 
        
        print_ph('              [//] Newton iterations for the non linear lithostatic pressure problem')

        # --- Newton =>         
        
        S.PL.x.array[:] = p_k.x.array[:]
      
        
        
        return S  
        
        
#-----------------------------------------------------------------
class Stokes_Problem(Problem):
    def __init__(self,M,elements,name):
        super().__init__(M,elements,name)       
        M = getattr(M,name[1])
        self.FSPT = dolfinx.fem.functionspace(M.mesh, elements[2]) 
        
    def set_newton():
        # Place holder -> Not finding useful tutorial for newton 
     
        pass
    
    def compute_shear_heating(self,ctrl,FR,S,D,sc,wedge=1):
        from .compute_material_property import compute_viscosity_FX
        from .utils import evaluate_material_property
        if wedge ==1: 
            V = S.u_wedge.function_space
            PT = self.FSPT
            e = compute_strain_rate(S.u_wedge)
            eta_new = compute_viscosity_FX(e, S.t_owedge, S.p_lwedge, FR, sc)
        else: 
            V = S.u_slab.function_space
            PT = self.FSPT
            e = compute_strain_rate(S.u_slab)
            eta_new = compute_viscosity_FX(e, S.t_oslab, S.p_lslab, FR, sc)

        shear_heating = ufl.inner(2*eta_new*e, e)

        if wedge ==1: 
            S.Hs_wedge = evaluate_material_property(shear_heating,PT)
        else: 
            S.Hs_slab = evaluate_material_property(shear_heating,PT)

        return S
    
    def compute_residuum_stokes(self, u_new, p_new, D, T, PL, FR, sc):
        V = u_new.function_space
        Q = p_new.function_space
        v = ufl.TestFunction(V)
        q = ufl.TestFunction(Q)

        e = compute_strain_rate(u_new)
        eta_new = compute_viscosity_FX(e, T, PL, FR,sc)
        f = fem.Constant(D.mesh, PETSc.ScalarType((0.0,) * D.mesh.geometry.dim))
        dx = ufl.dx

        Fmom = (ufl.inner(2*eta_new*ufl.sym(ufl.grad(u_new)), ufl.sym(ufl.grad(v))) * dx
                - ufl.inner(p_new, ufl.div(v)) * dx
                - ufl.inner(f, v) * dx)

        Fdiv = ufl.inner(q, ufl.div(u_new)) * dx

        Rm = fem.petsc.assemble_vector(fem.form(Fmom))
        Rm.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        if getattr(self, "bc", None):
            with Rm.localForm() as lf:
                r = lf.getArray(readonly=False)
                for bc in self.bc:
                    dofs = bc.dof_indices()[0]   # local dof indices
                    r[dofs] = 0.0

        rmom = Rm.norm()

        # divergence residual: I'd rely on divuL2 instead
        divuL2 = (fem.assemble_scalar(fem.form(ufl.inner(ufl.div(u_new), ufl.div(u_new))*dx)))**0.5

        # optional: keep your Q-vector residual if you really want it
        Rd = fem.petsc.assemble_vector(fem.form(Fdiv))
        Rd.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        rdiv = Rd.norm()

        return rmom, rdiv, divuL2

    
    
    def set_linear_picard(self,u_slab,T,PL,D,FR,ctrl,sc, a_p = None,it=0, ts = 0):
        
        """
        The problem is always LINEAR when depends on T/P -> Becomes fairly 
        non linear when the temperature is involved. 
        """
        
        u, p  = self.trial0, self.trial1
        v, q  = self.test0,  self.test1
        dx    = ufl.dx

        e = compute_strain_rate(u_slab)

        eta = compute_viscosity_FX(e,T,PL,FR,sc)

        a1 = ufl.inner(2*eta*ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * dx
        a2 = - ufl.inner(ufl.div(v), p) * dx             # build once
        a3 = - ufl.inner(q, ufl.div(u)) * dx             # build once
        a_p0 = ufl.inner(q, p) * dx                      # pressure mass (precond)

        f  = fem.Constant(D.mesh, PETSc.ScalarType((0.0,)*D.mesh.geometry.dim))
        f2 = fem.Constant(D.mesh, PETSc.ScalarType(0.0))
        L  = fem.form([ufl.inner(f, v)*dx, ufl.inner(f2, q)*dx])    
    
        return a1, a2, a3 , L , a_p0       

        

        
def L2_norm_calculation(f):
    return fem.assemble_scalar(fem.form(ufl.inner(f, f) * ufl.dx))**0.5       
        
        
#------------------------------------------------------------------
class Slab(Stokes_Problem): 
    def __init__(self,M:Mesh, elements:tuple, name:list,pdb:PhaseDataBase):
        super().__init__(M,elements,name)
    
    def setdirichlecht(self,ctrl,D,theta,it=0, ts = 0):
        mesh = self.F0.mesh 
        tdim = mesh.topology.dim
        fdim = tdim - 1
        
        # facet ids
        inflow_facets  = D.facets.find(D.bc_dict['inflow'])
        outflow_facets = D.facets.find(D.bc_dict['outflow'])
        slab_facets    = D.facets.find(D.bc_dict['top_subduction'])
        
        if (it == 0 or ts == 0 ):
            if ctrl.slab_bc == 1: # moving wall slab bc 
                # Compute this crap only once @ the beginning of the simulation 
                if it == 0 and ts == 0:

                        n = ufl.FacetNormal(mesh)                    # exact facet normal in    weak   form
                        v_slab = float(1.0)  # slab velocity magnitude

                        v_const = ufl.as_vector((ctrl.v_s[0], 0.0))

                        proj = ufl.Identity(mesh.geometry.dim) - ufl.outer(n, n)  # projector   onto  the  tangential plane
                        t = ufl.dot(proj, v_const)                     # tangential     velocity    vector on  slab
                        t_hat = t / ufl.sqrt(ufl.inner(t, t))    
                        v_project = v_slab * t_hat  # projected tangential velocity vector on   slab
                        self.moving_wall = fem.Function(self.F0)
                        w = self.trial0
                        v = self.test0
                        a = ufl.inner(w, v) * self.ds(D.bc_dict['top_subduction'])        #  boundary     mass    matrix (vector)
                        L = ufl.inner(v_project, v) * self.ds(D.bc_dict['top_subduction'])

                        self.moving_wall = fem.petsc.LinearProblem(
                            a, L,
                            petsc_options={
                            "ksp_type": "cg",
                            "pc_type": "jacobi",
                            "ksp_rtol": 1e-20,
                            }
                        ).solve()  # ut_h \in V
                self.moving_wall.x.array[:] = self.moving_wall.x.array[:]*ctrl.v_s[0] 

                dofs_s_x = fem.locate_dofs_topological(self.F0.sub(0), fdim, slab_facets)
                dofs_s_y = fem.locate_dofs_topological(self.F0.sub(1), fdim, slab_facets)

                bcx = fem.dirichletbc(self.moving_wall.sub(0), dofs_s_x)
                bcy = fem.dirichletbc(self.moving_wall.sub(1), dofs_s_y)
                return [bcx,bcy]
            else:
                # Pipe like boundary condition  {dirichlect bottom top and free slip in the boundaries applied with nietsche method.}
                dofs_in_x  = fem.locate_dofs_topological(self.F0.sub(0), fdim, inflow_facets)
                dofs_in_y  = fem.locate_dofs_topological(self.F0.sub(1), fdim, inflow_facets)
                dofs_out_x = fem.locate_dofs_topological(self.F0.sub(0), fdim, outflow_facets)
                dofs_out_y = fem.locate_dofs_topological(self.F0.sub(1), fdim, outflow_facets)
                
                # scalar BCs on subspaces
                bc_left_x   = fem.dirichletbc(PETSc.ScalarType(ctrl.v_s[0]), dofs_in_x,  self.F0.sub(0))
                bc_left_y   = fem.dirichletbc(PETSc.ScalarType(ctrl.v_s[1]), dofs_in_y,    self.F0.sub(1))
                bc_bottom_x = fem.dirichletbc(PETSc.ScalarType(ctrl.v_s[0]*np.cos(theta)),          dofs_out_x, self.F0.sub(0))
                bc_bottom_y = fem.dirichletbc(PETSc.ScalarType(ctrl.v_s[0]*np.sin(theta)),          dofs_out_y, self.F0.sub(1))
                
                return [bc_left_x,bc_left_y,bc_bottom_x,bc_bottom_y]
                
    
    def compute_nitsche_FS(self,D, S, dS, a1,a2,a3,FGS,gamma,sc,it = 0):
        """
        Compute the Nitsche free slip boundary condition for the slab problem.
        This is a placeholder function and should be implemented with the actual Nitsche    method.
        """
        def tau(eta, u):
            return 2 * eta * ufl.sym(ufl.grad(u))
        
        
        # Linear 
        e   = compute_strain_rate(S.u_slab)   
        eta = compute_viscosity_FX(e,S.t_oslab,S.p_lslab,FGS,sc)
        
        n = ufl.FacetNormal(D.mesh)
        h = ufl.CellDiameter(D.mesh)

        a1 += (
            - ufl.inner(tau(eta, self.trial0), ufl.outer(ufl.dot(self.test0, n) * n, n)) * self.ds(dS)
            - ufl.inner(ufl.outer(ufl.dot(self.trial0, n) * n, n), tau(eta, self.test0)) * self.ds(dS)
            + (2 * eta * gamma / h)
            * ufl.inner(ufl.outer(ufl.dot(self.trial0, n) * n, n), ufl.outer(self.test0, n))
            * self.ds(dS)
        )
        if it == 0:
            a2 += ufl.inner(self.trial1, ufl.dot(self.test0, n)) * self.ds(dS)
            a3 += ufl.inner(self.test1, ufl.dot(self.trial0, n)) * self.ds(dS)
        else:
            a2 += 0 
            a3 += 0 
        return a1, a2, a3 
    
    def Solve_the_Problem(self,S,ctrl,FGS,M,g,sc,it=0,ts=0):
        theta = M.g_input.theta_out_slab
        M = getattr(M,'domainA')
        
        
        """
        According to chatgpt, and I have already verified this notion before asking, 
        I cannot create test and trial function from a previous instance. In case of mixed space
        I need always to recreate on the spot, which is very annoying despite they are pretty much cheap 
        in terms of computation. 
        
    
        """
        V_subs0 = self.FS.sub(0)
        p_subs0 = self.FS.sub(1)
        V_subs, _ = V_subs0.collapse()
        p_subs, _ = p_subs0.collapse()
    
        self.trial0 = ufl.TrialFunction(V_subs)
        self.test0 = ufl.TestFunction(V_subs)
        self.trial1 = ufl.TrialFunction(p_subs)
        self.test1 = ufl.TestFunction(p_subs)
        # Create the linear problem
        a1,a2,a3, L, a_p = self.set_linear_picard(S.u_slab,S.t_oslab,S.p_lslab,M,FGS,ctrl,sc)

        # Create the dirichlecht boundary condition 
        self.bc   = self.setdirichlecht(ctrl,M,theta) 
        # Set Nietsche FS boundary condition 
        dS_top = M.bc_dict["top_subduction"]
        dS_bot = M.bc_dict["bot_subduction"]
        # 1 Extract ds 
        a1,a2,a3 = self.compute_nitsche_FS(M, S, dS_bot, a1, a2 ,a3,FGS ,50.0,sc)
        
        if ctrl.slab_bc == 0: 
            a1,a2,a3 = self.compute_nitsche_FS(M, S, dS_top, a1, a2 ,a3,FGS ,50.0,sc)
        
        # Slab problem is ALWAYS LINEAR
        
        a   = fem.form([[a1, a2],[a3, None]])
        a_p0  = fem.form([[a1, None],[None, a_p]])


        self.solv = SolverStokes(a, a_p0,L ,MPI.COMM_WORLD, 0,self.bc,self.F0,self.F1,ctrl ,J = None, r = None,it = 0, ts = 0)
        
        print_ph('              // -- // --- SLAB STOKES PROBLEM // -- // --->')    
        time_A = timing.time()    
        if self.solv.direct_solver==1:
            x = self.solv.A.createVecLeft()
        else:
            x = self.solv.A.createVecRight()
        self.solv.ksp.solve(self.solv.b, x)
        if self.solv.direct_solver == 0: 
            xu = x.getSubVector(self.solv.is_u)
            xp = x.getSubVector(self.solv.is_p)
            u, p = fem.Function(self.F0), fem.Function(self.F1)
    
            u.x.array[:] = xu.array_r
            p.x.array[:] = xp.array_r
            p.x.scatter_forward()
            u.x.scatter_forward()
            S.u_slab.x.array[:] = u.x.array[:]
            S.p_slab.x.array[:] = p.x.array[:]
            S.u_slab.x.scatter_forward()
            S.p_slab.x.scatter_forward()
        else: 
            S.u_slab.x.array[:self.solv.offset] = x.array_r[:self.solv.offset]
            S.p_slab.x.array[: (len(x.array_r) - self.solv.offset)] = x.array_r[self.solv.offset:]
            S.u_slab.x.scatter_forward()
            S.p_slab.x.scatter_forward()
        

        time_B = timing.time()
        print_ph(f'              // -- // --- Solution of Stokes problem in {time_B-time_A:.2f} sec // -- // --->')
        print_ph(f'')

        return S 
    
#---------------------------------------------------------------------------------------------------       
class Wedge(Stokes_Problem): 
    def __init__(self,M:Mesh, elements:tuple, name:list,pdb:PhaseDataBase):
        super().__init__(M,elements,name)
        
        M = getattr(M,'domainB')

        comm = MPI.COMM_WORLD

        # Example: each rank has some local IDs
        local_ids = np.int32(M.phase.x.array[:])

        # Gather arrays from all processes
        all_ids = comm.allgather(local_ids)

        # Root (or all, since allgather) concatenates
        all_ids = np.concatenate(all_ids)

        # Compute global unique IDs
        unique_ids = np.unique(all_ids)

        non_linear_v = False 
        non_linear_T = False 

        
        for i in range(np.shape(unique_ids)[0]): 
            if pdb.option_eta[unique_ids[i]] > 1: 
                non_linear_v = True
            elif pdb.option_eta[unique_ids[i]] > 0:
                non_linear_T = True
    
            
        if non_linear_v == True:
            self.typology = 'NonlinearProblem'
        elif non_linear_T == True and non_linear_v==False: 
            self.typology = 'NonlinearProblemT'
        else:
            self.typology = 'LinearProblem'
        
        
            
    
    def setdirichlecht(self,ctrl,D,theta,V,g_input,it=0, ts = 0):
        mesh = V.mesh 
        tdim = mesh.topology.dim
        fdim = tdim - 1
        
    
        # facet ids
        slab_facets      = D.facets.find(D.bc_dict['slab'])
        
            
        # Compute this crap only once @ the beginning of the simulation 


        noslip = np.zeros(mesh.geometry.dim, dtype=PETSc.ScalarType)
        dofs_over = fem.locate_dofs_topological(V, fdim, D.facets.find(D.bc_dict['overriding']))

        bc_overriding = fem.dirichletbc(noslip, dofs_over, V)
        
        
        #------------------------------------------------------------------------

        n = ufl.FacetNormal(mesh)                    # exact facet normal in    weak   form
        v_slab = float(1.0)  # slab velocity magnitude

        v_const = ufl.as_vector((ctrl.v_s[0], 0.0))

        proj = ufl.Identity(mesh.geometry.dim) - ufl.outer(n, n)  # projector   onto  the  tangential plane
        t = ufl.dot(proj, v_const)                     # tangential     velocity    vector on  slab
        t_hat = t / ufl.sqrt(ufl.inner(t, t))    
        v_project = v_slab * t_hat  # projected tangential velocity vector on   slab
        moving_wall_wedge = fem.Function(V)
        w = self.trial0
        v = self.test0
        

              
        a = ufl.inner(w, v) * self.ds(D.bc_dict['slab'])        #  boundary     mass    matrix (vector)
        
        if ctrl.decoupling == 1:
            scaling = fem.Function(self.FSPT)
            scaling = decoupling_function(self.FSPT.tabulate_dof_coordinates()[:,1],scaling,g_input)  
            scaling.x.array[:] = 1-scaling.x.array[:]
            scaling.x.scatter_forward()
            L = ufl.inner(v_project * scaling, v) * self.ds(D.bc_dict['slab'])
        else: 
            L = ufl.inner(v_project, v) * self.ds(D.bc_dict['slab'])

        moving_wall_wedge = fem.petsc.LinearProblem(
            a, L,
            petsc_options={
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1e-20,
            }
        ).solve()
            
        self.moving_wall_wedge = moving_wall_wedge.copy()
        self.moving_wall_wedge.x.array[:] = moving_wall_wedge.x.array[:]*ctrl.v_s[0] 

        dofs_s_x = fem.locate_dofs_topological(V.sub(0), fdim,slab_facets)
        dofs_s_y = fem.locate_dofs_topological(V.sub(1), fdim,slab_facets)

        bcx = fem.dirichletbc(self.moving_wall_wedge.sub(0), dofs_s_x)
        bcy = fem.dirichletbc(self.moving_wall_wedge.sub(1), dofs_s_y)
        
    
        return [bcx, bcy, bc_overriding]
                
   
    
    def Solve_the_Problem(self,S,ctrl,FGW,M,g,sc,g_input,it=0,ts=0):
        theta = M.g_input.theta_out_slab
        M = getattr(M,'domainB')

        
        """
        According to chatgpt, and I have already verified this notion before asking, 
        I cannot create test and trial function from a previous instance. In case of mixed space
        I need always to recreate on the spot, which is very annoying despite they are pretty much cheap 
        in terms of computation. 
        
    
        """
        V_subs0 = self.FS.sub(0)
        p_subs0 = self.FS.sub(1)
        V_subs, _ = V_subs0.collapse()
        p_subs, _ = p_subs0.collapse()
    
        self.trial0 = ufl.TrialFunction(V_subs)
        self.test0 = ufl.TestFunction(V_subs)
        self.trial1 = ufl.TrialFunction(p_subs)
        self.test1 = ufl.TestFunction(p_subs)
    

        # Create the linear problem
        a1,a2,a3, L, a_p = self.set_linear_picard(S.u_wedge,S.t_owedge,S.p_lwedge,M,FGW,ctrl,sc)

        # Create the dirichlecht boundary condition 
        self.bc   = self.setdirichlecht(ctrl,M,theta,V_subs,g_input) 
        
        # Form the system 
        a   = [[a1, a2],[a3, None]]
        a_p0  = [[a1, None],[None, a_p]]

        print_ph('              // -- // --- STOKES PROBLEM [WEDGE] // -- // --- > ')

        
        time_A = timing.time()

        
        if self.typology == 'LinearProblem' or self.typology == 'NonlinearProblemT': 
            S,r_al = self.solve_linear_picard(fem.form(a),fem.form(a_p0),fem.form(L),ctrl, S)

        else: 
            print_ph('              [//] Picard iterations for the non linear lithostatic pressure problem')

            u_k   = S.u_wedge.copy()
            p_k   = S.p_wedge.copy()
            du0   = S.u_wedge.copy()
            dp0   = S.p_wedge.copy()
            du1   = S.u_wedge.copy()
            dp1   = S.p_wedge.copy()
            
            res  = 1.0 
            it_inner   = 0 
            while (res > ctrl.tol_innerPic) and it_inner < ctrl.it_max: 
                time_ita = timing.time()
                if it_inner>0: 
                    a1,_,_, _,_ = self.set_linear_picard(u_k,S.t_owedge,S.p_lwedge,M,FGW,ctrl,sc)
                    a[0][0] = a1 
                    a       = fem.form(a)
                    a_p0[0][0] = a1 
                    a_p0       = fem.form(a_p0)
                
                S,r_al = self.solve_linear_picard(fem.form(a),fem.form(a_p0),fem.form(L),ctrl, S,it)
                
                
                du0.x.array[:]  = S.u_wedge.x.array[:] - u_k.x.array[:];du0.x.scatter_forward()
                
                du1.x.array[:] = S.u_wedge.x.array[:] + u_k.x.array[:];du1.x.scatter_forward()
                
                dp0.x.array[:]  = S.p_wedge.x.array[:] - p_k.x.array[:];dp0.x.scatter_forward()
                
                dp1.x.array[:] = S.p_wedge.x.array[:] + p_k.x.array[:];dp1.x.scatter_forward()
        
                rmom, rdiv, divuL2 = self.compute_residuum_stokes(S.u_wedge,S.p_wedge,M,S.t_owedge,S.p_lwedge,FGW,sc)
                
                if it_inner == 0:
                    rmom_0 = rmom
                    rdiv_0 = rdiv
                    r_al0 = r_al
                
        
                tol_u = L2_norm_calculation(du0)/L2_norm_calculation(du1)
                
                tol_p = L2_norm_calculation(dp0)/L2_norm_calculation(dp1)
                                
                res   = np.sqrt(tol_u**2+tol_p**2)
                
                u_k.x.array[:] = ctrl.relax*S.u_wedge.x.array[:] + (1-ctrl.relax)*u_k.x.array[:]
                p_k.x.array[:] = ctrl.relax*S.p_wedge.x.array[:] + (1-ctrl.relax)*p_k.x.array[:]

                time_itb = timing.time()

                print_ph(f'              []Wedge L_2 norm is   {res:.3e}, it_th {it_inner:d} performed in {time_itb-time_ita:.2f} seconds')
                print_ph(f'                         [x] |F^mom|/|F^mom_0| {rmom/rmom_0:.3e}, |F^div|/|F^div_0| {rdiv/rdiv_0:.3e}')
                print_ph(f'                         [x] |F^mom|           {rmom:.3e},         |F^div| {rdiv:.3e}')
                print_ph(f'                         [x] |r_al|/|r_al0|    {r_al/r_al0:.3e},         |r_al|  {r_al:.3e}')

                it_inner = it_inner+1 
        
            print_ph(f'              []Wedge L_2 norm is   {res:.3e}, it_th {it_inner:d} performed in {time_itb-time_ita:.2f} seconds')
            print_ph(f'                         [?] |F^mom|/|F^mom_0| {rmom/rmom_0:.3e}, |F^div|/|F^div_0| {rdiv/rdiv_0:.3e}')
            print_ph(f'                         [?] |F^mom|           {rmom:.3e}, abs div residuum |F^div| {rdiv:.3e}')
            print_ph('              []Converged ')

        time_B = timing.time()
        print_ph(f'              // -- // --- Solution of Wedge in {time_B-time_A:.2f} sec // -- // --- >')
        print_ph('')

        return S 
    
    def solve_linear_picard(self,a,a_p0,L,ctrl, S,it=0,ts=0):
            

        self.solv = SolverStokes(a, a_p0,L ,MPI.COMM_WORLD, 0,self.bc,self.F0,self.F1,ctrl ,J = None, r = None,it = 0, ts = 0)
        
        
        if self.solv.direct_solver==1:
            x = self.solv.A.createVecLeft()
        else:
            x = self.solv.A.createVecRight()
        self.solv.ksp.solve(self.solv.b, x)
        if self.solv.direct_solver == 0: 
            xu = x.getSubVector(self.solv.is_u)
            xp = x.getSubVector(self.solv.is_p)
    

            S.u_wedge.x.array[:] = xu.array_r[:]
            S.p_wedge.x.array[:] = xp.array_r[:]
            S.u_wedge.x.scatter_forward()
            S.p_wedge.x.scatter_forward()
            

            
        else: 
            S.u_wedge.x.array[:self.solv.offset] = x.array[:self.solv.offset]
            S.p_wedge.x.array[: (len(x.array_r) - self.solv.offset)] = x.array[self.solv.offset:]
            S.u_wedge.x.scatter_forward()
            S.p_wedge.x.scatter_forward()
        
        r = self.solv.b.copy()
        self.solv.A.mult(x, r)          # r = A x
        r.scale(-1.0)         # r = -A x
        r.axpy(1.0, self.solv.b)        # r = b - A x
        r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        abs_res = r.norm()



        return S,abs_res 

def compute_adiabatic_initial_adiabatic_contribution(M,T,Tgue,p,FG,vankeken): 
    
    from .compute_material_property import alpha_FX 
    from .utils import evaluate_material_property
    
    
    FS = T.function_space 
    Tg = fem.Function(FS)
    Tg = T.copy()
    v  = ufl.TestFunction(FS)


    if vankeken==0:
        
        res = 1
        while res > 1e-6:
        
            expr = (alpha_FX(FG,Tg,p) * p)/(heat_capacity_FX(FG,Tg) * density_FX(FG,Tg,p))
            a = T * ufl.exp(expr)
            TG1 = evaluate_material_property(a,FS)
            res = compute_residuum(TG1,Tg)
            Tg.x.array[:]  = 0.8*(TG1.x.array[:])+(1-0.8)*Tg.x.array[:]
            

            
        
    else: 
        from .utils import evaluate_material_property
        expr = (alpha_FX(FG,Tg,p) * p)/(heat_capacity_FX(FG,Tg) * density_FX(FG,Tg,p))
        F = T * ufl.exp(expr)
        Tg = evaluate_material_property(F,FS)

    
    return Tg
    
    

def initial_adiabatic_lithostatic_thermal_gradient(sol,lps,FGpdb,M,g,it_outer,ctrl):
    res = 1 
    it = 0
    T_0 = sol.T_O.copy()
    T_Oa = sol.T_O.copy()
    while res > 1e-3: 
        P_old = sol.PL.copy()
        sol = lps.Solve_the_Problem(sol,ctrl,FGpdb,M,g,it_outer,ts=0)
        T_O = compute_adiabatic_initial_adiabatic_contribution(M.domainG,T_0,T_Oa,sol.PL,FGpdb,ctrl.van_keken)
        resp = compute_residuum(sol.PL,P_old)
        resT = compute_residuum(T_O, T_Oa)
        res = np.sqrt(resp**2 + resT**2)
        if it !=0: 
            sol.PL.x.array[:] = 0.8 * sol.PL.x.array[:] + (1-0.8) * P_old.x.array[:]
            sol.PL.x.scatter_forward()
            T_Oa.x.array[:]= T_O.x.array[:] * 0.8 + (1-0.8) * T_O.x.array[:]
            sol.T_O.x.scatter_forward()
        it = it + 1 
        sol.T_O = T_Oa.copy()
        print_ph('Adiabatic res is %.3e'%res)
    
    sol.T_i = T_O.copy()
    sol.T_N = T_O.copy()
    sol.T_O = T_O.copy()
    return sol 

    

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
    
    from .thermal_structure_ocean import compute_initial_LHS
    from .compute_material_property import populate_material_properties_thermal,populate_material_properties_rheology

    
    element_p           = M.element_p#basix.ufl.element("Lagrange","triangle", 1) 
    
    element_PT          = M.element_PT#basix.ufl.element("Lagrange","triangle",2)
    
    element_V           = M.element_V#basix.ufl.element("Lagrange","triangle",2,shape=(2,))

    #==================== Phase Parameter ====================
    lhs_ctrl = compute_initial_LHS(ctrl,lhs_ctrl, sc, pdb,M.g_input.theta_in_slab)  
          
    # Define Problem
    # Global energy
    energy_global               = Global_thermal (M = M, name = ['energy','domainG']  , elements = (element_PT,                   ), pdb = pdb, ctrl = ctrl)
    # Global lithostatic pressure
    lithostatic_pressure_global = Global_pressure(M = M, name = ['pressure','domainG'], elements = (element_PT,                     ), pdb = pdb                                ) 
    # Wedge stokes problem
    wedge                       = Wedge          (M = M, name = ['stokes','domainB'  ], elements = (element_V,element_p,element_PT  ), pdb = pdb                                )
    # Slab stokes problem
    slab                        = Slab           (M = M, name = ['stokes','domainA'  ], elements = (element_V,element_p,element_PT  ), pdb = pdb                                )
    # Gravity, as I do not know where to put -> most likely inside the global problem 
    g = fem.Constant(M.domainG.mesh, PETSc.ScalarType([0.0, -ctrl.g]))    

    # Define Solution 
    # Create instance of solution.
    sol                         = Solution()
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
        PL_kouter       = sol.PL.copy()
        u_global_kouter = sol.u_global.copy()
        p_global_kouter = sol.p_global.copy()
        
        if (ctrl.adiabatic_heating != 0) and (it_outer==0) and (ts==0) :
            sol = initial_adiabatic_lithostatic_thermal_gradient(sol,
                                                                 LG,
                                                                 FGT,
                                                                 M,
                                                                 g,
                                                                 it_outer,
                                                                 ctrl)
        
        
        if LG.typology == 'NonlinearProblem' or it_outer == 0:  
            LG.Solve_the_Problem(sol,
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
            Sl.Solve_the_Problem(sol,
                                   ctrl
                                   ,FGSR
                                   ,M
                                   ,g
                                   ,sc,
                                   it = it_outer,
                                   ts=ts)

        if (We.typology == 'NonlinearProblem') or (We.typology == 'NonlinearProblemT') or (it_outer == 0):  
            We.Solve_the_Problem(sol
                                    ,ctrl
                                    ,FGWR
                                    ,M
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
        
        
        EG.Solve_the_Problem(sol
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
        sol = outerloop_operation(M,ctrl,ioctrl,sc,lhs,FGT,FGWR,FGSR,FGGR,EG,LG,We,Sl,sol,g,ts=ts)

        if ctrl.adiabatic_heating==0:
            sol.T_ad = compute_adiabatic_initial_adiabatic_contribution(M.domainG,sol.T_N,None,sol.PL,FGT,0)

        if ctrl.steady_state == 1 or (ts%10) == 0:
            output_class.print_output(sol,M.domainG,FGT,FGGR,ioctrl,sc,ctrl,it_outer=0,time=t*t*sc.T/sc.scale_Myr2sec,ts=ts)
        
        
        if ctrl.steady_state == 1: 
            t = ctrl.time_max
            if ctrl.van_keken == 1: 
                from .output import _benchmark_van_keken
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
            relative_slab_T_difference(sol)


        t = t+ctrl.dt
        
        sol.T_O = sol.T_N
        
        ts = ts + 1

    return 0 

# Def Outer iteration routine


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

def compute_residuum(a,b):
    dx = a.copy()
    
    dx1 = a.copy()
    
    dx.x.array[:]  = b.x.array[:] - a.x.array[:];dx.x.scatter_forward()
    
    dx1.x.array[:]  = b.x.array[:] + a.x.array[:];dx1.x.scatter_forward()
    
    
    res = L2_norm_calculation(dx)/L2_norm_calculation(dx1)
    
    return res

#------------------------------------------------------------------------------------------------------------

def min_max_array(a, vel = False):
    
    if vel == True: 
        a1 = a.sub(0).collapse()
        a2 = a.sub(1).collapse()
        array = np.sqrt(a1.x.array[:]**2 + a2.x.array[:]**2)
    else: 
        array = a.x.array[:]
    
    local_min = np.min(array[:])
    local_max = np.max(array[:])
    
    global_min = a.function_space.mesh.comm.allreduce(local_min, op=MPI.MIN)
    global_max = a.function_space.mesh.comm.allreduce(local_max, op=MPI.MAX)
    
    return np.array([global_min, global_max],dtype=np.float64)

#------------------------------------------------------------------------------------------------------------
def compute_residuum_outer(sol,T,PL,u,p,it_outer,sc,tA,Tmax,ts):
    # Prepare the variables 

    
    
    
    res_u = compute_residuum(sol.u_global,u)
    res_p = compute_residuum(sol.p_global,p)
    res_T = compute_residuum(sol.T_N,T)
    res_PL= compute_residuum(sol.PL,PL)
    
    minMaxU = min_max_array(sol.u_global, vel=True)
    minMaxP = min_max_array(sol.p_global)
    minMaxT = min_max_array(sol.T_N)
    minMaxPL= min_max_array(sol.PL)
    
    # scal back 
    
    minMaxU = minMaxU*(sc.L/sc.T)/sc.scale_vel 
    minMaxP = minMaxP*sc.stress/1e9 
    minMaxT = minMaxT*sc.Temp -273.15
    minMaxPL= minMaxPL*sc.stress/1e9
    
    if minMaxT[1]-(Tmax * sc.Temp-273.15)>1.0: 
        print_ph('Temperature higher than the maximum temperature')
    
    res_total = np.sqrt(res_u**2+res_p**2+res_T**2)
    if not np.isfinite(res_total):
        raise ValueError("res_total is NaN/Inf; check inputs and residual computations.")
    
    time_B_outer = timing.time()

    print_ph('')
    print_ph(f' Outer iteration {it_outer:d} with tolerance {res_total:.3e}, in {time_B_outer-tA:.1f} sec // -- // --->')
    print_ph(f'    []Res velocity       =  {res_u:.3e} [n.d.], max= {minMaxU[1]:.6f}, min= {minMaxU[0]:.6f} [cm/yr]')
    print_ph(f'    []Res Temperature    =  {res_T:.3e} [n.d.], max= {minMaxT[1]:.6f}, min= {minMaxT[0]:.6f} [C]')
    print_ph(f'    []Res pressure       =  {res_p:.3e} [n.d.], max= {minMaxP[1]:.3e}, min= {minMaxP[0]:.3e} [GPa]')
    print_ph(f'    []Res lithostatic    =  {res_PL:.3e}[n.d.], max= {minMaxPL[1]:.3e}, min= {minMaxPL[0]:.3e} [GPa]')
    print_ph(f'    []Res total (sqrt(rp^2+rT^2+rPL^2+rv^2)) =  {res_total:.3e} [n.d.] ')
    print_ph('. =============================================// -- // --->')
    print_ph('')

    sol.mT = np.append(sol.mT,minMaxT[0])
    sol.MT = np.append(sol.MT,minMaxT[1])
    sol.outer_iteration = np.append(sol.outer_iteration,res_total)
    sol.ts = np.append(sol.ts,ts)
    
    
    return res_total, sol 
#------------------------------------------------------------------------------------------------------------
def relative_slab_T_difference(sol):
    # Prepare the variables 

    

    res_T = compute_residuum(sol.t_nslab,sol.t_oslab)

    print_ph(f'    []dT slab   =  {res_T:.3e} [n.d.]')


    
    
    
    
    return 0
#------------------------------------------------------------------------------------------------------------
