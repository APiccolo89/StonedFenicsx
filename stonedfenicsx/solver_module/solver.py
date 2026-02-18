from stonedfenicsx.utils import *
from stonedfenicsx.package_import import *
from stonedfenicsx.scal import Scal
from dolfinx.fem.petsc          import assemble_matrix_block, assemble_vector_block


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
            "A", "P", "b",
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
    def __init__(self,A ,b ,COMM, nl, J = None, r = None):
        self.A = fem.petsc.create_matrix(fem.form(A)) # Store the sparsisity 
        self.b = fem.petsc.create_vector(fem.form(b)) # Store the vector
        self.ksp = PETSc.KSP().create(COMM)           # Create the ksp object 
        self.ksp.setOperators(self.A)                # Set Operator
        self.ksp.setType("gmres")
        self.ksp.setTolerances(rtol=1e-8, atol=1e-9)
        self.pc = self.ksp.getPC()
        self.pc.setType("lu")
        self.pc.setFactorSolverType("mumps")

        self.J = None
        self.r = None 
    
class SolverStokes(Solvers): 

    
    def __init__(self,a,a_p ,L ,COMM, nl,bcs ,F0,F1, ctrl,J = None, r = None,it = 0, ts = 0):
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
            self.pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
            self.pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)
        else: 
            self.update_block_operator(a,a_p,bcs,L,F0,F1)
            

    
    def set_iterative_solver(self,a,a_p ,L ,COMM, nl,bcs ,F0,F1, ctrl,J = None, r = None,it = 0, ts = 0): 
        #Return block operators and block RHS vector for the Stokes problem'
        # nullspace vector [0_u; 1_p] locally
        if it == 0 or ts == 0:
            self.set_block_operator(a,a_p,bcs,L,F0,F1)
            V_map = F0.dofmap.index_map
            Q_map = F1.dofmap.index_map
        
            offset_u = V_map.local_range[0] * F0.dofmap.index_map_bs + \
                   Q_map.local_range[0]
            offset_p = offset_u + V_map.size_local * F0.dofmap.index_map_bs
            is_u = PETSc.IS().createStride(V_map.size_local * F0.dofmap.index_map_bs, offset_u, 1,
            comm=PETSc.COMM_SELF)
            is_p = PETSc.IS().createStride(
            Q_map.size_local, offset_p, 1, comm=PETSc.COMM_SELF)
            
            
            self.offset_u = V_map.local_range[0] * F0.dofmap.index_map_bs + \
                   Q_map.local_range[0]
            self.offset_p = self.offset_u + V_map.size_local * F0.dofmap.index_map_bs
            self.is_u = is_u
            self.is_p = is_p

            
            assert self.nloc_u + self.nloc_p == self.A.getLocalSize()[1]
            
            self.ksp = PETSc.KSP().create(COMM)
            self.ksp.setOperators(self.A, self.P)
            self.ksp.setTolerances(rtol=1e-9)
            self.ksp.setType("fgmres")
            self.ksp.getPC().setType("fieldsplit")
            self.ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)
            self.ksp.getPC().setFieldSplitIS(("u", self.is_u), ("p", self.is_p))

            # Configure velocity and pressure sub-solvers
            self.ksp_u, self.ksp_p = self.ksp.getPC().getFieldSplitSubKSP()
            self.ksp_u.setType("preonly")
            self.ksp_u.getPC().setType("hypre")
            self.ksp_p.setType("preonly")
            self.ksp_p.getPC().setType("bjacobi")

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

        self.A = assemble_matrix_block(a, bcs=bcs)   ; self.A.assemble()
        self.P = assemble_matrix_block(a_p, bcs=bcs) ; self.P.assemble()
        self.b = assemble_vector_block(L, a, bcs=bcs); self.b.assemble()
        self.null_vec = self.A.createVecLeft()
        self.nloc_u = F0.dofmap.index_map.size_local * F0.dofmap.index_map_bs
        self.nloc_p = F1.dofmap.index_map.size_local
        self.null_vec.array[:self.nloc_u]  = 0.0
        self.null_vec.array[self.nloc_u:self.nloc_u+self.nloc_p] = 1.0
        self.null_vec.normalize()
        self.nsp = PETSc.NullSpace().create(vectors=[self.null_vec])
        self.A.setNullSpace(self.nsp)


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
            self.b, L, a, bcs=bcs)
        
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                   mode=PETSc.ScatterMode.REVERSE)
    
        # -------------------------
        # KSP operators
        # -------------------------
        if self.direct_solver == 0:
            # Nullspace (e.g., pressure) — set once if possible, but safe here too
            if getattr(self, "nsp", None) is not None:
                self.A.setNullSpace(self.nsp)

            self.ksp.setOperators(self.A, self.P)

            # Debug (optional)
            # if MPI.COMM_WORLD.rank == 0:
            #     print("A Fro norm:", self.A.norm(PETSc.NormType.FROBENIUS))

        else:
            # Direct solver: do NOT call ksp.setUp() every timestep (expensive)
            self.ksp.setOperators(self.A)
