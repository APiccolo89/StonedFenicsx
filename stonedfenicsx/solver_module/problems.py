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
    dx        : ufl.measure.Measure
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
            self.F0,_       = self.FS.sub(0).collapse()
            self.F1,_       = self.FS.sub(1).collapse()
            # Define trial/test on mixed FS
            self.trial0   = ufl.TrialFunction(self.FS.sub(0).collapse()[0])
            self.trial1   = ufl.TrialFunction(self.FS.sub(1).collapse()[0])
            self.test0    = ufl.TestFunction(self.FS.sub(0).collapse()[0])
            self.test1    = ufl.TestFunction(self.FS.sub(1).collapse()[0])
        
        self.dx       = ufl.Measure("dx", domain=M.mesh)
        self.ds       = ufl.Measure("ds", domain=M.mesh, subdomain_data=M.facets) # Exterior -> for boundary external 
        self.dS       = ufl.Measure("dS", domain=M.mesh, subdomain_data=M.facets) # Interior -> for boundary integral inside
        
        
#------------------------------------------------------------------
class Global_thermal(Problem):
    def __init__(self,M:Mesh, elements:tuple, name:list,pdb:PhaseDataBase,ctrl:NumericalControls):
        super().__init__(M,elements,name)
                
        self.steady_state = ctrl.steady_state
        
        if np.all(pdb.option_rho<2) and np.all(pdb.option_k==0) and np.all(pdb.option_Cp==0):
            self.typology = 'LinearProblem'
        else:
            self.typology = 'NonlinearProblem'
        
    def create_bc_temp(self,M:Mesh,ctrl:NumericalControls,geom,lhs,u_global,T_i,it,ts=0):
        from scipy.interpolate import griddata   
        cd_dof = self.FS.tabulate_dof_coordinates()
        # This part can be done only once -> bc dofs are constant 
        if ts == 0 and it == 0:
            facets                 = M.facets.find(M.bc_dict['Top'])    
            dofs_top               = fem.locate_dofs_topological(self.FS, M.mesh.topology.dim-1, facets)
            # -> Probably I need to use some parallel shit here 
            self.bc_top            = fem.dirichletbc(ctrl.Ttop, dofs_top, self.FS) 
            facets                 = M.facets.find(M.bc_dict['Left_inlet'])    
            dofs_left              = fem.locate_dofs_topological(self.FS, M.mesh.topology.dim-1, facets)
            
            # Create suitable function space for the problem
            T_bc_L = fem.Function(self.FS)
            # Extract z and lhs 
            z   = lhs.z
            LHS = lhs.LHS 
            # Extract coordinate dofs
            # Interpolate temperature field: 
            T_bc_L.x.array[:] = griddata(z, LHS, cd_dof[:,1], method='nearest')
            T_bc_L.x.scatter_forward()
            self.bc_left = fem.dirichletbc(T_bc_L, dofs_left)
            

            facets                 = M.facets.find(M.bc_dict['Right_lit'])                        
            dofs_right_lit        = fem.locate_dofs_topological(self.FS, M.mesh.topology.dim-1, facets)
    
            T_gr = (-geom.lab_d-0)/(ctrl.Tmax-ctrl.Ttop)
            T_gr = T_gr**(-1) 
        
            bc_fun = fem.Function(self.FS)
            bc_fun.x.array[dofs_right_lit] = ctrl.Ttop + T_gr * cd_dof[dofs_right_lit,1]
            bc_fun.x.scatter_forward()
        
            self.bc_right_lit = fem.dirichletbc(bc_fun, dofs_right_lit)

        facets                 = M.facets.find(M.bc_dict['Right_wed'])                        
        dofs_right_wed        = fem.locate_dofs_topological(self.FS, M.mesh.topology.dim-1, facets)
        h_vel  = u_global.sub(0) # index 1 = y-direction (2D)
        vel_T  = fem.Function(self.FS)
        vel_T.interpolate(h_vel)
        vel_bc = vel_T.x.array[dofs_right_wed]
        ind_z = np.where((vel_bc < 0.0) & (cd_dof[dofs_right_wed,1]<=-geom.lab_d))
        dofs_vel = dofs_right_wed[ind_z[0]]        
        if ctrl.adiabatic_heating==0:
            self.bc_right_wed = fem.dirichletbc(ctrl.Tmax, dofs_vel,self.FS)
        else: 
            function_bc = fem.Function(self.FS)
            function_bc.interpolate(T_i)
            self.bc_right_wed  = fem.dirichletbc(function_bc,dofs_vel)
            
        facets                 = M.facets.find(M.bc_dict['Bottom_wed'])                        
        dofs_bot_wed        = fem.locate_dofs_topological(self.FS, M.mesh.topology.dim-1, facets)
        h_vel       = u_global.sub(0) # index 1 = y-direction (2D)   
        v_vel       = u_global.sub(1) # index 1 = y-direction (2D)
        vel_T  = fem.Function(self.FS)
        vel_T.interpolate(v_vel)
        vel_bc = vel_T.x.array[dofs_bot_wed]
        ind_z = np.where((vel_bc > 0.0))
        dofs_vel = dofs_bot_wed[ind_z[0]]                
        
        self.bc_bot_wed = fem.dirichletbc(ctrl.Tmax, dofs_vel,self.FS)
        
        bc = [self.bc_top, self.bc_left,  self.bc_right_wed,self.bc_bot_wed, self.bc_right_lit]

        
        
        return bc 
        
    #------------------------------------------------------------------
    def compute_shear_heating(self,ctrl,pdb,S,D,g_input,sc):
        """
        Apperently the sociopath the devise this method, uses a delta function to describe 
        the interface frictional heating. 
        -> [A] => Shear heating becomes a ufl expression. So happy about it 
        
        """

        if ctrl.decoupling == 1 and ctrl.model_shear >0: 
            facets1                = D.facets.find(D.bc_dict['Subduction_top_lit'])
            facets2                = D.facets.find(D.bc_dict['Subduction_top_wed'])

            facet_seismogenic = np.unique(np.concatenate((facets1,facets2)))

            dofs              = fem.locate_dofs_topological(self.FS, D.mesh.topology.dim-1, facet_seismogenic)

            heat_source = fem.Function(self.FS)
            heat_source.x.array[:] = 0.0   
            heat_source.x.scatter_forward()
        
            decoupling    = heat_source.copy()
            Z = self.FS.tabulate_dof_coordinates()[:,1]
            decoupling = decoupling_function(Z,decoupling,g_input)

            if ctrl.model_shear==2:
                # compute the plastic strain rate ratio and viscous shear heating strain rate 
                # Place holder function
                expression = self.compute_friction_shear_expression(pdb,ctrl,D,S.T_O,S.PL,ctrl.v_s[0],decoupling,sc,dofs) * ufl.avg(self.test0) * (self.dS(D.bc_dict['Subduction_top_lit']) +self.dS(D.bc_dict['Subduction_top_wed']))

            else:  
                phi = np.tan(pdb.friction_angle)
                expression = decoupling * ufl.avg(S.PL) * ctrl.v_s[0] * phi * ufl.avg(self.test0) * (self.dS(D.bc_dict['Subduction_top_lit']) +self.dS(D.bc_dict['Subduction_top_wed']))

            return expression

        else:
            return 0.0 
        
        
    def compute_friction_shear_expression(self,pdb,ctrl,D,T,P,vs,decoupling,sc,dofs):

        from .compute_material_property import compute_plastic_strain

        e_II_fr = (vs * decoupling * 1 /ctrl.wz_tk)/2  # Second invariant strain rate

        # -> compute the plastic strain rate

        e_pl, tau = compute_plastic_strain(e_II_fr,T,P,pdb,D.phase,ctrl.phase_wz-1,sc)

        e_vs = 1 - e_pl 

        phi = ufl.tan(pdb.friction_angle)

        friction = (e_pl * vs * decoupling * phi * P + e_vs * e_II_fr * tau * ctrl.wz_tk) 

        return friction 
            
    
    def set_newton_SS(self,p,D,T,u_global,pdb):
        trial0 = self.trial0                # δp (TrialFunction) for Jacobian


        rho_k = density_FX(pdb, T, p, D.phase, D.mesh)  # frozen
        
        k_k = density_FX(pdb, T, p, D.phase, D.mesh)  # frozen
        
        Cp_k = heat_capacity_FX(pdb, T,  D.phase, D.mesh)  # frozen


        f    = fem.Constant(D.mesh, 0.0)  # source term
        
        dx  = self.dx
        # Linear operator with frozen coefficients
            
        diff = ufl.inner(k_k * ufl.grad(T), ufl.grad(self.test0)) * dx
        adv  = rho_k *Cp_k *ufl.dot(u_global, ufl.grad(T)) * self.test0 * dx

        a_lin = diff + adv
        L     = f * self.test0 * dx   
        # Nonlinear residual: F(p; v) = ∫ ∇p·∇v dx - ∫ ∇v·(ρ(T, p) g) dx
        F = a_lin - L
        # Jacobian dF/dp in direction δp (trial0)
        J = ufl.derivative(F, T, trial0)
        
        return F,J 
    
    #------------------------------------------------------------------
    def compute_energy_source(self,D,FG):
        source = fem.Function(self.FS)
        source = compute_radiogenic(FG, source)
        self.energy_source = source.copy()

    #------------------------------------------------------------------
    def compute_adiabatic_heating(self,D,FG,u,T,p,ctrl):
        from .compute_material_property import alpha_FX
        
        if ctrl.adiabatic_heating != 0: 
            
        
            alpha = alpha_FX(FG,T,p)
            adiabatic_heating = alpha * T * ufl.inner(ufl.grad(p), u) 
        else: 
            adiabatic_heating = (0.0)
        
        
        self.adiabatic_heating = adiabatic_heating
        

    #------------------------------------------------------------------
    def compute_residual_SS(self,p_k,T,u_global,D,FG, ctrl):
        # Function that set linear form and linear picard for picard iteration
        
        rho_k = density_FX(FG, T, p_k)  # frozen
        
        Cp_k = heat_capacity_FX(FG, T)  # frozen

        k_k = heat_conductivity_FX(FG, T, p_k, Cp_k, rho_k)  # frozen

        self.compute_adiabatic_heating(D,FG,u_global,T,p_k,ctrl)

        f    = self.energy_source# source term
        
        dx  = self.dx
        # Linear operator with frozen coefficients
            
        diff = ufl.inner(k_k * ufl.grad(T), ufl.grad(self.test0)) * dx
        
        adv  = rho_k * Cp_k *ufl.dot(u_global, ufl.grad(T)) * self.test0 * dx
            
        L = fem.form((f + self.adiabatic_heating) * self.test0 * dx + self.shear_heating )      
        
        R = fem.form(diff + adv - L)
                

        return R
    
    
    def set_linear_picard_SS(self,
                             p_k:dolfinx.fem.Function=None,
                             T:dolfinx.fem.Function = None,
                             T_O:dolfinx.fem.Function=None,
                             u_global:dolfinx.fem.Function=None,
                             D:Domain =None,
                             FG:Functions_material_properties_global=None,
                             ctrl:NumericalControls=None,
                             dt:float = None,
                             it:int=0)->tuple[dolfinx.fem.Form,dolfinx.fem.Form]:
        # Function that set linear form and linear picard for picard iteration
        
        rho_k = density_FX(FG, T, p_k)  # frozen
        
        Cp_k = heat_capacity_FX(FG, T)  # frozen

        k_k = heat_conductivity_FX(FG, T, p_k, Cp_k, rho_k)  # frozen


        f    = self.energy_source# source term
        
        dx  = self.dx
        # Linear operator with frozen coefficients
        if it == 0: 
            
            diff = ufl.inner(k_k * ufl.grad(self.trial0), ufl.grad(self.test0)) * dx
            
            adv  = rho_k * Cp_k *ufl.dot(u_global, ufl.grad(self.trial0)) * self.test0 * dx
            
            a = fem.form(diff + adv)
            
            L = fem.form((f) * self.test0 * dx + self.shear_heating )      
        
        else: 
        
            a = fem.Form(None)
                

        return a, L
    #------------------------------------------------------------------

    def set_linear_picard_TD(self,
                             p_k:dolfinx.fem.Function=None,
                             T:dolfinx.fem.Function = None,
                             T_O:dolfinx.fem.Function=None,
                             u_global:dolfinx.fem.Function=None,
                             D:Domain =None,
                             FG:Functions_material_properties_global=None,
                             ctrl:NumericalControls=None,
                             dt:float = None,
                             it:int=0):
        # Function that set linear form and linear picard for picard iteration
        # Crank Nicolson scheme 
        # a - > New temperature 
        # L - > Old temperature
        # -> Source term is assumed constant in time and do not vary between the timesteps 
        
        rho_k = density_FX(FG, T, p_k)  # frozen
                
        Cp_k = heat_capacity_FX(FG, T)  # frozen

        k_k = heat_conductivity_FX(FG, T, p_k, Cp_k, rho_k)  # frozen


        
        rho_k0 = density_FX(FG, T_O, p_k)  # frozen
                
        Cp_k0 = heat_capacity_FX(FG, T_O)  # frozen
        
        k_k0 = heat_conductivity_FX(FG, T_O, p_k, Cp_k, rho_k)  # frozen


                
        rhocp        =  (rho_k * Cp_k)

        rhocp_old    =  (rho_k0 * Cp_k0)
        
        dx  = self.dx
        
        f    = (self.energy_source) * self.test0 * dx + self.shear_heating # source term {energy_source is radiogenic heating compute before hand, shear heating is frictional heating already a form}

        
        # a -> New temperature 
        diff_new = ( 1 / 2 ) * ufl.inner(k_k * ufl.grad(self.trial0), ufl.grad(self.test0)) * dx
        
        adv_new  = (rhocp / 2 )* ufl.dot(u_global, ufl.grad(self.trial0)) * self.test0 * dx
        
        mass_new = (rhocp / dt) * self.trial0 * self.test0 * dx
        
        a = fem.form(diff_new + adv_new + mass_new)
                
        if it == 0: 
            
            adv_old =  - (rhocp_old / 2 ) * ufl.dot(u_global, ufl.grad(T_O)) * self.test0 * dx

            diff_old =  - ( 1 / 2 ) * ufl.inner(k_k0 * ufl.grad(T_O), ufl.grad(self.test0)) * dx
            
            mass_old =  (rhocp_old / dt) * T_O * self.test0 * dx
            
            L = fem.form(diff_old + adv_old + f + mass_old)
            
            return a, L

        else: 
            return a, fem.Form(None)
    #------------------------------------------------------------------
    def Solve_the_Problem(self,S,ctrl,FG,M,lhs,geom,sc,it=0,ts=0): 
        
        nl = 0 
        dt = None
        # choose the problem: 
        if ctrl.steady_state == 1: 
            self.set_linear = self.set_linear_picard_SS 
        else: 
            self.set_linear = self.set_linear_picard_TD
            dt = ctrl.dt
        
        
        p_k = S.PL.copy()  # Previous lithostatic pressure 
        T   = S.T_N # -> will not eventually update 
        
            
        if it == 0:         
            self.shear_heating = self.compute_shear_heating(ctrl,FG, S,getattr(M,'domainG'),geom,sc)
            self.compute_energy_source(getattr(M,'domainG'),FG)
        
        a,L = self.set_linear(p_k
                              ,T
                              ,S.T_O
                              ,S.u_global
                              ,getattr(M,'domainG')
                              ,FG
                              ,ctrl
                              ,dt)
        
        self.bc = self.create_bc_temp(getattr(M,'domainG'),ctrl,geom,lhs,S.u_global,S.T_i,it)

        if it == 0 & ts == 0: 
            self.solv = ScalarSolver(a,L,M.comm,nl)
        
        print_ph('              // -- // --- Temperature problem [GLOBAL] // -- // --->')
        
        time_A = timing.time()

        if nl == 0: 
            S.T_N = self.solve_the_linear(S,a,L,S.T_N) 
        else: 
            S = self.solve_the_non_linear(M,S,Hs,ctrl,pdb)
        
        time_B = timing.time()
        
        print_ph(f'              // -- // --- Solution of Temperature  in {time_B-time_A:.2f} sec // -- // --->')



        return S 

    #------------------------------------------------------------------

    def Solve_the_Problem_SS(self,S,ctrl,FG,M,lhs,geom,sc,it=0,ts=0): 
        
        nl = 0 
        
        # choose the problem: 

        
        
        p_k = S.PL.copy()  # Previous lithostatic pressure 
        T   = S.T_N # -> will not eventually update 
        
        if ctrl.adiabatic_heating ==2:
            Hs = S.Hs_global # Shear heating
        else:
            Hs = 0.0
            
        if it == 0:         
            self.shear_heating = self.compute_shear_heating(ctrl,FG, S,getattr(M,'domainG'),geom,sc)
            self.compute_energy_source(getattr(M,'domainG'),FG)
        
        a,L = self.set_linear(p_k,T,S.u_global,Hs,getattr(M,'domainG'),FG,ctrl)
        
        self.bc = self.create_bc_temp(getattr(M,'domainG'),ctrl,geom,lhs,S.u_global,S.T_i,it)
        if self.typology == 'NonlinearProblem':
            F,J = self.set_newton_SS(p_k,getattr(M,'domainG'),T,S.u_global,pdb)
            nl = 1 
        else: 
            F = None;J=None 

        if it == 0 & ts == 0: 
            self.solv = ScalarSolver(a,L,M.comm,nl,J,F)
        
        print_ph('              // -- // --- Temperature problem [GLOBAL] // -- // --->')
        
        time_A = timing.time()

        if nl == 0: 
            S = self.solve_the_linear(S,a,L,S.T_O) 
        else: 
            S = self.solve_the_non_linear_SS(M,S,Hs,ctrl,pdb)
        
        time_B = timing.time()
        
        print_ph(f'              // -- // --- Solution of Temperature  in {time_B-time_A:.2f} sec // -- // --->')



        return S 
    #------------------------------------------------------------------
    
    def Solve_the_Problem_TD(self,S,ctrl,pdb,M,lhs,geom,sc,it=0,ts=0): 

        nl = 0 
        p_k = S.PL.copy()  # Previous lithostatic pressure 



        if it == 0:
         
            self.shear_heating = self.compute_shear_heating(ctrl,pdb, S,getattr(M,'domainG'),geom,sc)
            self.compute_energy_source(getattr(M,'domainG'),pdb)


        a,L = self.set_linear_picard_TD(p_k,S.T_N,S.T_O,S.u_global,getattr(M,'domainG'),pdb,ctrl.dt)

        self.bc = self.create_bc_temp(getattr(M,'domainG'),ctrl,geom,lhs,S.u_global,it)
        if self.typology == 'NonlinearProblem':
            nl = 1 
        F = None 
        J = None 

        if it == 0 & ts == 0: 
            self.solv = ScalarSolver(a,L,M.comm,nl,J,F)
        

        print_ph('     // -- // --- Temperature problem [GLOBAL] // -- // --->')
        
        time_A = timing.time()

        if nl == 0: 
            S.T_N = self.solve_the_linear(S,a,L,S.T_O) 
        else: 
            S = self.solve_the_non_linear_TD(M,S,ctrl,pdb)
        
        time_B = timing.time()
        
        print_ph(f'. // -- // --- Solution of Temperature  in {time_B-time_A:.2f} sec // -- // --->')



        return S 
    
    def solve_the_linear(self,S,a,L,fen_function,isPicard=0,it=0,ts=0):
        

        self.solv.A.zeroEntries()
        fem.petsc.assemble_matrix(self.solv.A,fem.form(a),self.bc)
        self.solv.A.assemble()
        # b -> can change as it is the part that depends on the pressure in case of nonlinearities
        self.solv.b.set(0.0)
        fem.petsc.assemble_vector(self.solv.b, fem.form(L))
        fem.petsc.apply_lifting(self.solv.b, [fem.form(a)], [self.bc])
        self.solv.b.ghostUpdate()
        fem.petsc.set_bc(self.solv.b, self.bc)
        self.solv.ksp.solve(self.solv.b, fen_function.x.petsc_vec)
        
        if isPicard == 0: # if it is a picard iteration the function gives the function 
            fen_function.x.scatter_forward()
            return fen_function 
        else:
            return fen_function


    def solve_the_non_linear(self
                            ,M
                            ,S
                            ,Hs
                            ,ctrl
                            ,FGT
                            ,it=0):  
        
        isPicard = 1 # Flag for the linear solver. 
        tol = 1.0 
        T_O = S.T_O
        T_k = S.T_N.copy() 
        T_k1 = S.T_N.copy()
        du   = S.PL.copy()
        du2  = S.PL.copy()
        it_inner = 0 
        time_A = timing.time()
        print_ph('              [//] Picard iterations for the non linear temperature problem')

        while it_inner < max_it and tol > ctrl.tol:
            time_ita = timing.time()
            
            if it_inner == 0: 
                A,L = self.set_linear_picard(S.PL
                                            ,T_k
                                            ,S.u_global
                                            ,Hs,getattr(M,'domainG')
                                            ,pdb
                                            ,ctrl)
            else: 
                if ctrl.steady_state==1: 
                    _,L = self.set_linear_picard(S.PL
                                                ,T_k
                                                ,S.u_global
                                                ,Hs,getattr(M,'domainG')
                                                ,pdb
                                                ,ctrl
                                                ,it=it_inner)
                else: 
                    A,_ = self.set_linear_picard(S.PL
                             ,T_k
                             ,S.u_global
                             ,Hs,getattr(M,'domainG')
                             ,pdb
                             ,ctrl
                             ,it_inner)
                    
            T_k1 = self.solve_the_linear(S,A,L,T_k1,1,it,1)
            T_k1.x.scatter_forward()
            # L2 norm 
            du.x.array[:]  = T_k1.x.array[:] - T_k.x.array[:];du.x.scatter_forward()
            du2.x.array[:] = T_k1.x.array[:] + T_k.x.array[:];du2.x.scatter_forward()
            tol= L2_norm_calculation(du)/L2_norm_calculation(du2)
            
            time_itb = timing.time()
            print_ph(f'              []Temperature L_2 norm is {tol:.3e}, it_th {it_inner:d} performed in {time_itb-time_ita:.2f} seconds')
            
            #update solution
            T_k.x.array[:] = T_k1.x.array[:]*0.7 + T_k.x.array[:]*(1-0.7)
            
            it_inner = it_inner + 1 
        S.T_N.x.array[:] = T_k1.x.array[:]
        S.T_N.scatter_forward()
        print_ph(f'')
        
        return S  
        
    @timing_function
    def initial_temperature_field(self,M, ctrl, lhs, g_input):
        from scipy.interpolate import griddata
        from ufl import conditional, Or, eq
        from functools import reduce
        """
        X    -:- Functionspace (i.e., an abstract stuff that represents all the possible solution for the given mesh and element type)
        M    -:- Mesh object (i.e., a random container of utils related to the mesh)
        ctrl -:- Control structure containing the information of the simulations 
        lhs  -:- left side boundary condition controls. Separated from the control structure for avoiding clutter in the main ctrl  
        ---- 
        Function: Create a function out of the function space (T_i). From the function extract dofs, interpolate (initial) lhs all over. 
        Then select the crustal+lithospheric marker, and overwrite the T_i with a linear geotherm. Simple. 
        ----
        output : T_i the initial temperature field.  
            T_gr = (-M.g_input.ns_depth-0)/(ctrl.Tmax-ctrl.Ttop)
            T_gr = T_gr**(-1) 

            bc_fun = fem.Function(X)
            bc_fun.x.array[dofs_dirichlet] = ctrl.Ttop + T_gr * cd_dof[dofs_dirichlet,1]
            bc_fun.x.scatter_forward()
        """    
        #- Create part of the thermal field: create function, extract dofs, 
        X     = self.FS
        T_i_A = fem.Function(X)
        cd_dof = X.tabulate_dof_coordinates()
        T_i_A.x.array[:] = griddata(lhs.z, lhs.LHS, cd_dof[:,1], method='nearest')
        T_i_A.x.scatter_forward() 
        #- 
        T_gr = (-g_input.lab_d-0)/(ctrl.Tmax-ctrl.Ttop)
        T_gr = T_gr**(-1) 
        T_expr = fem.Function(X)
        ind_A = np.where(cd_dof[:,1] >= -g_input.lab_d)[0]
        ind_B = np.where(cd_dof[:,1] < -g_input.lab_d)[0]
        T_expr.x.array[ind_A] = ctrl.Ttop + T_gr * cd_dof[ind_A,1]
        T_expr.x.array[ind_B] = ctrl.Tmax
        T_expr.x.scatter_forward()
        T_i = fem.Function(X)
        expr = conditional(
            reduce(Or,[eq(M.phase, i) for i in [2, 3, 4, 5]]),
            T_expr,
            T_i_A
        )
        T_i.interpolate(fem.Expression(expr, X.element.interpolation_points()))
        T_i.x.array[ind_B] = ctrl.Tmax
        return T_i 
#-----------------------------------------------------------------