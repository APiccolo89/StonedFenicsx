from stonedfenicsx.utils import *
from dolfinx.fem.petsc          import assemble_matrix_block, assemble_vector_block
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem
"""This section has been created using the default tutorials of FEniCSx. 
I choose to use monolithic direct/iterative solvers. The choiche of the options depends
on the preconditioner that I set up for the stokes for the given problem. 
"""

class Solvers():
    def destroy(self):
        """Explicitly free PETSc objects held by this solver."""
        # Destroy in roughly reverse order of dependency.
        for name in (
            # KSPs first (they reference operators/PC)
            "ksp_u", "ksp_p", "ksp",
            # Index sets
            "is_u", "is_p",
            # Nullspace + vector
            "nsp", "null_vec",
            # Matrices / vectors
            "A", "P", "b", "x",
        ):
            obj = getattr(self, name, None)
            if obj is None:
                continue
            try:
                obj.destroy()
            except Exception:
                # Some PETSc wrappers may already be destroyed; ignore safely.
                pass
            finally:
                setattr(self, name, None)

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
    def __init__(self,a ,L, bcs ,COMM,direct =0):
        """Create the PETSc matrix/vector/KSP for a scalar (temperature or lithostatic pressure) problem.

        Allocates the system matrix `A` (with the sparsity of `a`) and RHS
        vector `b`, creates the KSP, and configures either an iterative
        (FGMRES + hypre) or direct (LU via MUMPS) solver.

        Args:
            a (dolfinx.fem.Form | ufl.Form): Bilinear form defining the matrix sparsity/operator.
            L (dolfinx.fem.Form | ufl.Form): Linear form defining the RHS vector.
            bcs (list): Dirichlet boundary conditions (unused directly here,
                kept for interface parity with callers).
            COMM (mpi4py.MPI.Comm): MPI communicator for the KSP.
            direct (int, optional): 0 for the iterative (fgmres/hypre)
                solver, non-zero for the direct (LU/mumps) solver. Defaults to 0.
        """
        self.A = fem.petsc.create_matrix(fem.form(a)) # Store the sparsisity
        self.b = fem.petsc.create_vector(fem.form(L)) # Store the vector
        self.ksp = PETSc.KSP().create(COMM)           # Create the ksp object 
        self.ksp.setOperators(self.A)                # Set Operator
        direct_solver = direct 
        if direct_solver == 0: 
        
            self.ksp.setType("fgmres")
            self.ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=2000)
            self.pc = self.ksp.getPC()
            self.pc.setType("hypre")
        else: 
            self.ksp.setType("preonly")
            self.pc = self.ksp.getPC()
            self.pc.setType("lu")       
            self.pc.setFactorSolverType('mumps')     

    
class SolverStokes(Solvers): 

    
    def __init__(self,a,a_p ,L ,COMM, nl,bcs ,F0,F1, ctrl,J = None, r = None,it = 0, ts = 0,slab=0):
        """Create the block PETSc solver for a Stokes (velocity-pressure) problem.

        Dispatches to `set_direct_solver` or `set_iterative_solver` based on
        `ctrl.stokes_solver_type` (always direct for the slab domain, whose
        internal-wall boundary condition over-constrains the problem).

        Args:
            a (dolfinx.fem.Form): Compiled block system form (momentum + continuity).
            a_p (dolfinx.fem.Form): Compiled preconditioner (pressure mass) block form.
            L (dolfinx.fem.Form): Compiled block RHS form.
            COMM (mpi4py.MPI.Comm): MPI communicator for the KSP.
            nl (int): Nonlinear-solve flag, forwarded to the solver setup (unused directly here).
            bcs (list): Dirichlet boundary conditions.
            F0 (dolfinx.fem.FunctionSpace): Collapsed velocity subspace (used
                for computing the velocity/pressure dof offset).
            F1 (dolfinx.fem.FunctionSpace): Collapsed pressure subspace.
            ctrl (NumericalControls): Numerical controls; `stokes_solver_type`
                selects direct (1) vs. iterative (0).
            J (optional): Unused; kept for interface parity. Defaults to None.
            r (optional): Unused; kept for interface parity. Defaults to None.
            it (int, optional): Outer iteration index, forwarded to the solver setup. Defaults to 0.
            ts (int, optional): Timestep index, forwarded to the solver setup. Defaults to 0.
            slab (int, optional): 1 forces the direct solver (slab domain). Defaults to 0.
        """
        if slab == 1:
            self.direct_solver = 1
        else:
            self.direct_solver = ctrl.stokes_solver_type
        self.offset = F0.dofmap.index_map.size_local * F0.dofmap.index_map_bs
        if self.direct_solver == 1: 
            self.set_direct_solver(a,a_p ,L ,COMM, nl,bcs ,F0,F1, ctrl,J = None, r = None,it=it, ts = ts)
        elif self.direct_solver ==0: 
            self.set_iterative_solver(a,a_p ,L ,COMM, nl,bcs ,F0,F1, ctrl,J = None, r = None,it = it, ts = ts)    
    
    def set_direct_solver(self,
                          a,
                          a_p,
                          L,
                          COMM, 
                          nl,
                          bcs,
                          F0,
                          F1, 
                          ctrl,
                          J = None, 
                          r = None,
                          it = 0,
                          ts = 0):
        """Configure a direct (LU/MUMPS) block solver for the Stokes system.

        On the first solve (`it == 0 and ts == 0`) builds the block
        operators/preconditioner and creates a `preonly` KSP with an LU/MUMPS
        preconditioner; on subsequent calls just reassembles the block
        operators in place via `update_block_operator`.

        Args:
            a (dolfinx.fem.Form): Compiled block system form.
            a_p (dolfinx.fem.Form): Compiled preconditioner block form.
            L (dolfinx.fem.Form): Compiled block RHS form.
            COMM (mpi4py.MPI.Comm): MPI communicator for the KSP.
            nl (int): Unused; kept for interface parity.
            bcs (list): Dirichlet boundary conditions.
            F0 (dolfinx.fem.FunctionSpace): Collapsed velocity subspace.
            F1 (dolfinx.fem.FunctionSpace): Collapsed pressure subspace.
            ctrl (NumericalControls): Numerical controls (unused directly here).
            J (optional): Unused; kept for interface parity. Defaults to None.
            r (optional): Unused; kept for interface parity. Defaults to None.
            it (int, optional): Outer iteration index. Defaults to 0.
            ts (int, optional): Timestep index. Defaults to 0.
        """
        if it == 0 and ts == 0:
            self.set_block_operator(a, a_p, bcs, L, F0, F1)

            # Create a solver
            self.ksp = PETSc.KSP().create(COMM)
            self.ksp.setOperators(self.A)
            self.ksp.setType("preonly")

            # Set the solver type to MUMPS (LU solver) and configure MUMPS to
            # handle pressure nullspace
            self.pc = self.ksp.getPC()
            self.pc.setType("lu")
            self.pc.setFactorSolverType("mumps")
            self.pc.setFactorSetUpSolverType()
            #self.pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
            #self.pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)
        else: 
            self.update_block_operator(a,a_p,bcs,L,F0,F1)
            

    
    def set_iterative_solver(self,a,a_p ,L ,COMM, nl,bcs ,F0,F1, ctrl,J = None, r = None,it = 0, ts = 0):
        """Configure an iterative (FGMRES + fieldsplit) block solver for the Stokes system.

        On the first solve (`it == 0 or ts == 0`) builds the block operators/
        preconditioner, computes the velocity/pressure PETSc index sets
        (`is_u`/`is_p`) for field-split, and creates an FGMRES KSP with a
        multiplicative fieldsplit preconditioner (hypre on each velocity and
        pressure sub-block). On subsequent calls just reassembles the block
        operators in place via `update_block_operator`.

        Args:
            a (dolfinx.fem.Form): Compiled block system form.
            a_p (dolfinx.fem.Form): Compiled preconditioner block form.
            L (dolfinx.fem.Form): Compiled block RHS form.
            COMM (mpi4py.MPI.Comm): MPI communicator for the KSP.
            nl (int): Unused; kept for interface parity.
            bcs (list): Dirichlet boundary conditions.
            F0 (dolfinx.fem.FunctionSpace): Collapsed velocity subspace.
            F1 (dolfinx.fem.FunctionSpace): Collapsed pressure subspace.
            ctrl (NumericalControls): Numerical controls; `iterative_solver_tol`
                sets the KSP relative tolerance.
            J (optional): Unused; kept for interface parity. Defaults to None.
            r (optional): Unused; kept for interface parity. Defaults to None.
            it (int, optional): Outer iteration index. Defaults to 0.
            ts (int, optional): Timestep index. Defaults to 0.
        """
        #Return block operators and block RHS vector for the Stokes problem'
        # nullspace vector [0_u; 1_p] locally
        if it == 0 or ts == 0:
            # Create the block operator and the pre-conditioner
            self.set_block_operator(a,a_p,bcs,L,F0,F1)
            
            # Map the dofs [Find the dofs that belongs to P, and the one that belongs to u]
            V_map = F0.dofmap.index_map
            Q_map = F1.dofmap.index_map
        
            # Rhs = [vu,vp]-> where do they start 
            # -> more over divided into the processors, so must find the layout for every processor
            offset_u = V_map.local_range[0] * F0.dofmap.index_map_bs + \
                   Q_map.local_range[0]
            offset_p = offset_u + V_map.size_local * F0.dofmap.index_map_bs
            # Petsc black magic to create the index array for recognising what node is u or p 
            is_u = PETSc.IS().createStride(V_map.size_local * F0.dofmap.index_map_bs, offset_u, 1,
            comm=PETSc.COMM_WORLD)
            is_p = PETSc.IS().createStride(
            Q_map.size_local, offset_p, 1, comm=PETSc.COMM_WORLD)
            
            # Save the information
            self.offset_u = V_map.local_range[0] * F0.dofmap.index_map_bs + \
                   Q_map.local_range[0]
            self.offset_p = self.offset_u + V_map.size_local * F0.dofmap.index_map_bs
            self.is_u = is_u
            self.is_p = is_p
            
            # Create the KSP object
            self.ksp = PETSc.KSP().create(COMM)
            # Set the operator 
            self.ksp.setOperators(self.A, self.P)
            # Set the tollerance of krylov solver
            self.ksp.setTolerances(rtol=ctrl.iterative_solver_tol)
            # Set the type fgmres (global minimum residual) -> Iterative solver, solve an approximate inversion of A-> and the residual is the 
            # the way to quantify how much the iterative solver is converged to real solution
            self.ksp.setType("fgmres")
            # 
            self.ksp.getPC().setType("fieldsplit")
            self.ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)
            self.ksp.getPC().setFieldSplitIS(("u", self.is_u), ("p", self.is_p))

            # Configure velocity and pressure sub-solvers
            self.ksp_u, self.ksp_p = self.ksp.getPC().getFieldSplitSubKSP()
            self.ksp_u.setType("preonly")
            self.ksp_u.getPC().setType("hypre")
            self.ksp_p.setType("preonly")
            self.ksp_p.getPC().setType("hypre")

            monitor_n_digits = int(np.ceil(np.log10(self.ksp.max_it)))
            def monitor(ksp, it, r):
                """Print the KSP iteration count and residual (PETSc KSP monitor callback).

                Args:
                    ksp (PETSc.KSP): The KSP instance (unused, required by the monitor signature).
                    it (int): Current iteration number.
                    r (float): Current (preconditioned) residual norm.
                """
                PETSc.Sys.Print(f"{         it: {monitor_n_digits}d}: {r:.3e}")

            #self.ksp.setMonitor(monitor)

            # The matrix A combined the vector velocity and scalar pressure
            # parts, hence has a block size of 1. Unlike the MatNest case, GAMG
            # cannot infer the correct near-nullspace from the matrix block
            # size. Therefore, we set block size on the top-left block of the
            # preconditioner so that GAMG can infer the appropriate near
            # nullspace.
            self.ksp.getPC().setUp()
            self.Pu, _ = self.ksp_u.getPC().getOperators()
            self.Pu.setBlockSize(2)
        else:
            self.update_block_operator(a,a_p,bcs,L,F0,F1)

    def set_block_operator(self,a,a_p,bcs,L,F0,F1):
        """Assemble the block system matrix, preconditioner matrix, RHS vector and pressure nullspace.

        Builds `self.A`/`self.P`/`self.b` from scratch via
        `assemble_matrix_block`/`assemble_vector_block`, then constructs the
        constant-pressure nullspace vector (zero on velocity dofs, uniform
        on pressure dofs, normalised) used for pure Neumann/free-slip
        pressure setups.

        Args:
            a (dolfinx.fem.Form): Compiled block system form.
            a_p (dolfinx.fem.Form): Compiled preconditioner block form.
            bcs (list): Dirichlet boundary conditions.
            L (dolfinx.fem.Form): Compiled block RHS form.
            F0 (dolfinx.fem.FunctionSpace): Collapsed velocity subspace (used
                to size the velocity block of the nullspace vector).
            F1 (dolfinx.fem.FunctionSpace): Collapsed pressure subspace (used
                to size the pressure block of the nullspace vector).
        """
        self.A = assemble_matrix_block(a, bcs=bcs)   ; self.A.assemble()
        self.P = assemble_matrix_block(a_p, bcs=bcs) ; self.P.assemble()
        self.b = assemble_vector_block(L, a, bcs=bcs); self.b.assemble()
        
        
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                   mode=PETSc.ScatterMode.FORWARD)

        # Solution vector, allocated once and reused by solve_linear_picard
        # every outer iteration instead of calling createVecLeft() per call.
        self.x = self.A.createVecLeft()
        self.null_vec = self.A.createVecLeft()
        self.nloc_u = F0.dofmap.index_map.size_local * F0.dofmap.index_map_bs
        self.nloc_p = F1.dofmap.index_map.size_local
        self.null_vec.array[:self.nloc_u]  = 0.0
        self.null_vec.array[self.nloc_u:self.nloc_u+self.nloc_p] = 1.0
        self.null_vec.normalize()
        self.nsp = PETSc.NullSpace().create(vectors=[self.null_vec])
        #self.A.setNullSpace(self.nsp)


    def update_block_operator(self,a,a_p,bcs,L,F0,F1):
        """
        Reassemble cached PETSc operators/vectors in-place.

        Assumes:
          - self.A, self.P, self.b were created once and match the block layout of (a, a_p, L)
          - FunctionSpaces / dofmaps used by a, a_p, L, and bcs are NOT recreated each timestep
        """

        # -------------------------
        # A (system / Jacobian)
        # -------------------------
        self.A.zeroEntries()
        assemble_matrix_block(self.A, a, bcs=bcs)
        self.A.assemble()

        # -------------------------
        # P (preconditioner matrix)
        # -------------------------
        self.P.zeroEntries()
        assemble_matrix_block(self.P, a_p, bcs=bcs)
        self.P.assemble()

        # -------------------------
        # b (RHS)
        # -------------------------
        with self.b.localForm() as b_u_local:
            b_u_local.set(0.0)
        
        dolfinx.fem.petsc.assemble_vector_block(
            self.b, L, a,bcs=bcs)
                

        # -------------------------
        # KSP operators
        # -------------------------
        if self.direct_solver == 0:
            # Nullspace (e.g., pressure) — set once if possible, but safe here too
            #if getattr(self, "nsp", None) is not None:
            #    self.A.setNullSpace(self.nsp)
                #self.nsp.remove(self.b)

            self.ksp.setOperators(self.A, self.P)

            # Debug (optional)
            # if MPI.COMM_WORLD.rank == 0:
            #     print("A Fro norm:", self.A.norm(PETSc.NormType.FROBENIUS))

        else:
            # Direct solver: do NOT call ksp.setUp() every timestep (expensive)
            self.ksp.setOperators(self.A)
