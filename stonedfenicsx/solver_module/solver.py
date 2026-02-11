from stonedfenicsx.utils import *
from stonedfenicsx.package_import import *
from stonedfenicsx.scal import Scal
from dolfinx.fem.petsc          import assemble_matrix_block, assemble_vector_block


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

        self.J = None
        self.r = None 
    
class SolverStokes(): 

    
    def __init__(self,a,a_p ,L ,COMM, nl,bcs ,F0,F1, ctrl,J = None, r = None,it = 0, ts = 0):
        self.direct_solver = 1
        if self.direct_solver == 1: 
            self.set_direct_solver(a,a_p ,L ,COMM, nl,bcs ,F0,F1, ctrl,J = None, r = None,it = 0, ts = 0)
            self.offset = F0.dofmap.index_map.size_local * F0.dofmap.index_map_bs
        elif self.direct_solver ==0: 
            self.set_iterative_solver(a,a_p ,L ,COMM, nl,bcs ,F0,F1, ctrl,J = None, r = None,it = 0, ts = 0)    
    
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
        
        if it == 0 or ts == 0:
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

    
    def set_iterative_solver(self,a,a_p ,L ,COMM, nl,bcs ,F0,F1, ctrl,J = None, r = None,it = 0, ts = 0): 
        #Return block operators and block RHS vector for the Stokes problem'
        # nullspace vector [0_u; 1_p] locally
        if it == 0 or ts == 0:
            self.set_block_operator(a,a_p,bcs,L,F0,F1)
            
            
            offset_u = 0
            offset_p = self.nloc_u
            # local starts in the *assembled block vector*

            self.is_u = PETSc.IS().createStride(self.nloc_u, offset_u, 1, comm=PETSc.COMM_SELF)
            self.is_p = PETSc.IS().createStride(self.nloc_p, offset_p, 1, comm=PETSc.COMM_SELF)
            V_map = F0.dofmap.index_map
            Q_map = F1.dofmap.index_map
            bs_u  = F0.dofmap.index_map_bs  # = mesh.dim for vector CG
            nloc_u = V_map.size_local * bs_u
            nloc_p = Q_map.size_local

            # local starts in the *assembled block vector*



            self.is_u = PETSc.IS().createStride(nloc_u, offset_u, 1, comm=PETSc.COMM_SELF)
            self.is_p = PETSc.IS().createStride(nloc_p, offset_p, 1, comm=PETSc.COMM_SELF)
            
            
            self.ksp = PETSc.KSP().create(COMM)
            self.ksp.setOperators(self.A, self.P)
            self.ksp.setTolerances(rtol=1e-9)
            self.ksp.setType("minres")
            self.ksp.getPC().setType("fieldsplit")
            self.ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
            self.ksp.getPC().setFieldSplitIS(("u", self.is_u), ("p", self.is_p))

            # Configure velocity and pressure sub-solvers
            self.ksp_u, self.ksp_p = self.ksp.getPC().getFieldSplitSubKSP()
            self.ksp_u.setType("preonly")
            self.ksp_u.getPC().setType("gamg")
            self.ksp_p.setType("preonly")
            self.ksp_p.getPC().setType("jacobi")

            # The matrix A combined the vector velocity and scalar pressure
            # parts, hence has a block size of 1. Unlike the MatNest case, GAMG
            # cannot infer the correct near-nullspace from the matrix block
            # size. Therefore, we set block size on the top-left block of the
            # preconditioner so that GAMG can infer the appropriate near
            # nullspace.
            self.ksp.getPC().setUp()
            self.Pu, _ = self.ksp_u.getPC().getOperators()
            self.Pu.setBlockSize(2)

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

        
