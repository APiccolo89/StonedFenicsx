import numpy as np
from reading_mesh.Mesh_c import *
import scipy.sparse as sps
import scipy.sparse.linalg.dsolve as linsolve
import time as timing
from dataclasses import dataclass
from solver_function.compute_material_property import viscosity
from solver_function.compute_material_property import compute_thermal_property
from numba import njit,prange
from numba.experimental import jitclass 
from numba import float64
from solver_function.numerical_control import NumericalControls
from reading_mesh.Mesh_c import MESH
from reading_mesh.Mesh_c import Computational_Data
from material_property.phase_db import PhaseDataBase
from solver_function.bc import bc_class
from scipy.sparse import coo_matrix
from solver_function.compute_material_property import density
from solver_function.scal import Scal
from petsc4py import PETSc

#import pypardiso as pypar
#-----------------------------------------------------------------------------------------------------------------------
velocity_spec = [
    ('u', float64[:]),
    ('v', float64[:]),
    ('u_old', float64[:]),
    ('v_old', float64[:]),
    ('area',float64[:]),
]

@jitclass(velocity_spec)
class VelocityFields:
    def __init__(self, nv,nel):
        self.u = np.zeros(nv)
        self.v = np.zeros(nv)
        self.u_old = np.zeros(nv)
        self.v_old = np.zeros(nv)
        self.area  = np.zeros(nel)


scalar_spec = [
    ('p', float64[:]),
    ('T', float64[:]),
    ('T_old', float64[:]),
    ('p_lit', float64[:]),
    ('area', float64[:])
]

@jitclass(scalar_spec)
class ScalarFields:
    def __init__(self, NfemP, NfemT,nel):
        self.p = np.zeros(NfemP)
        self.T = np.zeros(NfemT)
        self.T_old = np.zeros(NfemT)
        self.p_lit = np.zeros(NfemT)
        self.area = np.zeros(nel)

shape_spec = [
    ('NNNV', float64[:, :]),
    ('NNNP', float64[:, :]),
    ('NNNT', float64[:, :]),
    ('dNVdsr', float64[:,:,:]),
    ('dNTdsr', float64[:,:,:]),
]
#----------------------------------------------------------------------------------------------------------
@jitclass(shape_spec)
class ShapeFunctions:
    def __init__(self, mV, mP, mT, nquel):

        self.NNNV    = np.zeros((mV,nquel),dtype=np.float64)
        self.NNNP    = np.zeros((mP,nquel),dtype=np.float64)
        self.NNNT    = np.zeros((mT,nquel),dtype=np.float64)
        self.dNVdsr  = np.zeros((2,mV,nquel),dtype=np.float64)
        self.dNTdsr  = np.zeros((2,mT,nquel),dtype=np.float64)

quad_spec = [
    ('x_q', float64[:]),
    ('y_q', float64[:]),
    ('eta_q', float64[:]),
    ('T_q', float64[:]),
    ('exx_q', float64[:]),
    ('eyy_q', float64[:]),
    ('exy_q', float64[:]),
    ('k_q', float64[:]),
    ('rho_q', float64[:]),
    ('heatC_q', float64[:]),
    ('PL_q', float64[:]),
]
#-----------------------------------------------------------------------------------------------------------------------
@jitclass(quad_spec)
class QuadratureFields:
    def __init__(self, nq):
        self.x_q = np.zeros(nq)
        self.y_q = np.zeros(nq)
        self.eta_q = np.zeros(nq)
        self.T_q = np.zeros(nq)
        self.exx_q = np.zeros(nq)
        self.eyy_q = np.zeros(nq)
        self.exy_q = np.zeros(nq)
        self.k_q = np.zeros(nq)
        self.rho_q = np.zeros(nq)
        self.heatC_q = np.zeros(nq)
        self.PL_q = np.zeros(nq)


#-----------------------------------------------------------------------------------------------------------------------

#stokes_njit = False


@dataclass(slots=False)
class Sol:
    velocities: VelocityFields
    scalars: ScalarFields
    shapes: ShapeFunctions
    quadrature: QuadratureFields
    c_mat: np.ndarray

    def __init__(self, M,CD):
        nq = CD.nq * M.nel
        nquel = len(CD.qweights)

        # Initialize Velocity Fields
        self.velocities = VelocityFields(M.nv,M.nel)
          
          
        # Initialize Scalar Fields (Pressure, Temperature)
        self.scalars = ScalarFields(CD.NfemP, CD.NfemT,M.nel)
       

        # Initialize Shape Functions
        self.shapes = ShapeFunctions( CD.mV, CD.mP, CD.mT, 6)

        # Initialize Quadrature Fields (Numerical integration data)
        self.quadrature = QuadratureFields(nq)

        # Initialize constant matrix
    def _precompute_shape_functions(self,CD):
        vel_fem = field_fem(CD.vel_fem)
        pres_fem = field_fem(CD.pres_fem)
        temp_fem = field_fem(CD.temp_fem)

        for i in range(len(CD.qweights)):
            rq = CD.qcoords_r[i]
            sq = CD.qcoords_s[i]

            self.shapes.NNNV[:,i]    = vel_fem.NN(rq,sq)
            self.shapes.NNNP[:,i]    = pres_fem.NN(rq,sq)
            self.shapes.NNNT[:,i]    = temp_fem.NN(rq,sq)

            self.shapes.dNVdsr[0,:,i] = vel_fem.NNdr(rq,sq)
            self.shapes.dNVdsr[1,:,i] = vel_fem.NNds(rq,sq)
            self.shapes.dNTdsr[0,:,i] =  temp_fem.NNdr(rq,sq)
            self.shapes.dNTdsr[1,:,i] =  temp_fem.NNds(rq,sq)

        return self

#-----------------------------------------------------------------------------------------------------------------------

def scipy_sparse_to_petsc(A_csr):
    A_petsc = PETSc.Mat().createAIJ(
    size=A_csr.shape,
    csr=(A_csr.indptr, A_csr.indices, A_csr.data),
    comm=PETSc.COMM_WORLD)
    A_petsc.assemble()
    return A_petsc


def numpy_rhs_to_petsc(b_np):
    b_petsc = PETSc.Vec().createSeq(len(b_np), comm=PETSc.COMM_WORLD)
    b_petsc.setValues(range(len(b_np)), b_np)
    b_petsc.assemble()
    return b_petsc

@njit
def cotah(x):
    r = np.cosh(x)/np.sinh(x)
    return r 

#-----------------------------------------------------------------------------------------------------------------------
"""
Compute the lithostatic pressure field using Anthony Jourdon1,2 and Dave A. May1 2022. 
"""
#-----------------------------------------------------------------------------------------------------------------------


@njit
def _impose_litho_BC(BC,el_con,mT,K_el,f_el,rhs,row,col,val,idx):

    for k1 in range(0,mT):
        m1=el_con[k1]-1
        if BC.bc_pressure_fix[m1]:
           Aref=K_el[k1,k1]
           for k2 in range(0,mT):
               m2=el_con[k2]-1
               f_el[k2]-=K_el[k2,k1]*BC.bc_pressure_val[m1]
               K_el[k1,k2]=0
               K_el[k2,k1]=0
           K_el[k1,k1]=Aref
           f_el[k1]=Aref*BC.bc_pressure_val[m1]
        # end for
        # assemble matrix A_mat and right hand side rhs
    for k1 in range(0,mT):
        m1=el_con[k1]-1
        for k2 in range(0,mT):
            m2=el_con[k2]-1
            #A_mat[m1,m2]+=a_el[k1,k2]
            row[idx] = m1
            col[idx] = m2
            val[idx] = K_el[k1,k2]
            idx = idx + 1

        rhs[m1] = rhs[m1] + f_el[k1]

    return rhs,row,col,val,idx


#------------------------------------------------------------------------------------------------------------------------
#def loop_quadrature_lithostatic():

@njit
def volume_integral_lithostatic_p(M,Cq,CD,quad,T,PL,shapes,iel,K_el,f_el,mT,N_mat,B_mat,counterq,ctrl,pdb,it,sc):
    
    nq = CD.nq

    x   = M.x[M.el_con[iel,0:mT]-1]

    y   = M.y[M.el_con[iel,0:mT]-1]

    T   = T[M.el_con[iel,0:CD.mT]-1]

    PL  = PL[M.el_con[iel,0:CD.mT]-1]

    pres_sc = sc.stress

    temp_sc = sc.Temp

    rho_sc = sc.rho


    # Element coordinates 

    x   = M.x[M.el_con[iel,0:mT]-1]

    y   = M.y[M.el_con[iel,0:mT]-1]

    # -- 
    # X coordinate of the node

    X = np.zeros((2, mT), dtype=np.float64)

    X[0, :] = x
    X[1, :] = y
    
    # X matrix must be node x 2 
    X = np.transpose(X)  
    
    for kq in range(0,nq):


        weightq              = Cq[kq,2]
        # Shape function NT at the current quadrature point vector of mT entries 
        N_mat[0:CD.mT]        = shapes.NNNT[:,kq]

        # -
        # Shape function derivativies dNNNVdr and dNNNVds at the current quadrature point vector of mT entries  
        dN_mat = shapes.dNTdsr[:,:,kq]

        # Compute the jacobian and its inverse
        jcb = np.dot(dN_mat,X)  # 2x2 jacobian matrix
        # Compute the determinant and inverse of the jacobian
        jcob = np.linalg.det(jcb)  # determinant of the jacobian

        jcbi = np.linalg.inv(jcb)  # inverse of the jacobian
        
        # compute dNdx & dNdy - global derivatives of the shape functions

        dNX_mat = np.dot(jcbi,dN_mat)  # 2xnode matrix with dNdx and dNdy

        # - 
        ## compute diffusion matrix
        # Compute the thermal properties

        T_q = np.dot(N_mat,T)
        
        PL_q = np.dot(N_mat,PL)


        rho_q= density(ctrl.option_rho,T_q*temp_sc,PL_q*pres_sc,it)

        K_el = K_el + dNX_mat.T.dot(dNX_mat) * weightq * jcob

        g_vec = np.array([0,-ctrl.g],dtype=np.float64)

        f_el = f_el + weightq * jcob * (rho_q/rho_sc) * dNX_mat.T.dot(g_vec)

        counterq = counterq + 1 
        

    return K_el,f_el,counterq,quad
#-----------------------------------------------------------------------------------------------------------------------
#@njit
def create_stiffiness_lithostatic(M:MESH,
                             scalars:ScalarFields,
                             quad: QuadratureFields,
                             shapes: ShapeFunctions,
                             CD:Computational_Data,
                             BC:bc_class,
                             ctrl:NumericalControls,
                             pdb:PhaseDataBase,
                             it:int,
                             sc:Scal):

    #Computational data
    NfemT   = (CD.NfemT)
    mT     = CD.mT


    # Global level
    #A_sparse=lil_matrix((Nfem,Nfem),dtype=np.float64)
    rhs    = np.zeros(NfemT,dtype=np.float64)         # right hand side of Ax=b



    # Extract qcoord 
    rq = CD.qcoords_r
    sq = CD.qcoords_s
    wq = CD.qweights
    Cq = np.zeros((rq.shape[0], 3), dtype=np.float64)    
    Cq[:,0] = rq
    Cq[:,1] = sq
    Cq[:,2] = wq 

    # 
    #row = np.zeros((M.nel*),dtype=np.int64)

    counterq=0
    # Create the row column and val of the sparse matrix
    row = np.zeros((M.nel*36),dtype=np.int32)

    col =np.zeros((M.nel*36),dtype=np.int32)

    val = np.zeros((M.nel*36),dtype=np.float64)
    # counter for the sparse matrix
    idx = 0 
    for iel in prange(M.nel):
    
        B_mat = np.zeros((CD.ndim,CD.mT),dtype=np.float64)# gradient matrix B 
        N_mat = np.zeros((CD.mT),dtype=np.float64)# gradient matrix B 

                
        f_el=np.zeros(mT,dtype=np.float64)

        K_el = np.zeros((CD.mT,CD.mT),dtype=np.float64) # stiffness matrix K_el
        

        K_el,f_el,counterq,quad = volume_integral_lithostatic_p(M,Cq,CD,quad,scalars.T,scalars.p_lit,shapes,iel,K_el,f_el,mT,N_mat,B_mat,counterq,ctrl,pdb,it,sc)

        rhs,row,col,val,idx   = _impose_litho_BC(BC,M.el_con[iel,:],mT,K_el,f_el,rhs,row,col,val,idx)


    return rhs,quad,row,col,val
#------------------------------------------------------------------------------------------------------------------------
#def preamble_create_stiffiness_lithostatic():


def preamble_create_stiffiness_plith(M,scalars,quadrature,shapes,CD,BC,ctrl,pdb,it,Nfem,sc):


    
    rhs,quadrature,row,col,val = create_stiffiness_lithostatic(M,scalars,quadrature,shapes,CD,BC,ctrl,pdb,it,sc)

    A_sparse = coo_matrix((val, (row, col)), shape=(Nfem, Nfem))

    # 2. Remove duplicates (safe and smart)
    A_sparse.sum_duplicates()
    # 3. Eliminate zeros (optional, slight speedup if there are many zeros)
    A_sparse.eliminate_zeros()
    
    A_sparse = A_sparse.tocsr()

    if ctrl.petsc == 1: 
        # Convert to PETSc format if using PETSc
        A_sparse = scipy_sparse_to_petsc(A_sparse)
        rhs = numpy_rhs_to_petsc(rhs)


    return A_sparse, rhs, quadrature
#-----------------------------------------------------------------------------------------------------------------------
#def lithostatic_pressure():


def lithostatic_pressure(M:MESH,
                        S:Sol,
                        CD:Computational_Data,
                        BC:bc_class,
                        ctrl:NumericalControls,
                        pdb:PhaseDataBase,
                        it:int,
                        sc:Scal):
    
    
    
    # Solve lithostatic pressure field 
    print("    :::: Lithostatic pressure ::::    _")
    start_total = timing.time()
    Nfem  = (CD.NfemT)
    res = 1.0 
    p_lith = S.scalars.p_lit.copy()
    itp = 0 
    while res > 1e-3:
        start = timing.time()

        A_sparse,rhs,S.quadrature = preamble_create_stiffiness_plith(M,S.scalars,S.quadrature,S.shapes,CD,BC,ctrl,pdb,it,Nfem,sc)
        print("       ._.")
        print("      |   |Assembly in %.3f s, -> %.4f s per element" % ((timing.time() - start),(timing.time() - start)/M.nel))

        if ctrl.petsc == 1:
            x_petsc = rhs.duplicate()

            ksp = PETSc.KSP().create()
            ksp.setOperators(A_sparse)
            ksp.setType('preonly')  # Direct solve (no iteration)
            pc = ksp.getPC()
            pc.setType('lu')        # LU factorization
            pc.setFactorSolverType('mumps')  # or 'superlu', 'umfpack' if available
            ksp.setFromOptions()

            ksp.solve(rhs, x_petsc)
            sol = x_petsc.getArray()
        else:
            sol = sps.linalg.spsolve(A_sparse,rhs)
        
        S.scalars.p_lit = sol.T.copy()

        #np.savetxt('velocity.ascii',np.array([M.x,M.y,u,v]).T,header='# x,y,u,v')
        if itp>0:
           S.scalars.p_lit = ctrl.relax * S.scalars.p_lit + (1-ctrl.relax) * p_lith
   
        res = np.linalg.norm(p_lith-S.scalars.p_lit,2)/np.linalg.norm(p_lith+S.scalars.p_lit,2)
        print("      | %d |solve time: %.3f s" %(itp,timing.time() - start))
        p_lith = S.scalars.p_lit.copy()
        itp += 1
        print("      |   |[//]residual is : %.3e after %d iterations " %(res,itp))
        print("      |._.| " )

        print("    ---- ----    ")
    
    print("      [||]Total solve time: %.3f s" %(timing.time() - start_total))
    print("    :::: Lithostatic pressure ::::    |")
    return S.scalars.p_lit


#-----------------------------------------------------------------------------------------------------------------------

@njit
def compute_phase_ratio_numba(Phase_el: np.ndarray, phase_ratio: np.ndarray,
                               r: float, s: float, num_phases: int):
    
    rVnodes = np.array((0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0/3.))
    sVnodes = np.array((0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 1.0/3.))
    
    for i in range(num_phases):
        for ip in range(rVnodes.shape[0]):
            if Phase_el[ip] == i:
                d = np.sqrt((rVnodes[ip] - r) ** 2 + (sVnodes[ip] - s) ** 2)
                if d == 0:
                    d = 1e-12
                phase_ratio[i] += 1.0 / d

    total = np.sum(phase_ratio)
    phase_ratio /= total

    return phase_ratio

#------------------------------------------------------------------------------------------------------------------------
#@njit(nopython=stokes_njit)
def _impose_BC_stokes_pressure(el_conP,el_con,CD,BC,h_rhs,f_rhs,G_el,f_el,K_el,h_el,pressure_scaling,iel,row,col,val,idx,dir_P):
    Nfem  = CD.NfemV + CD.NfemP
    NfemV = CD.NfemV
    NfemP = CD.NfemP
    ndofV = CD.ndofV
    mV    = CD.mV
    mP    = CD.mP

    # --- Apply velocity Dirichlet BCs ---
    for k1 in range(mV):
        for i1 in range(ndofV):
            ikk = ndofV * k1 + i1
            m1  = ndofV * (el_con[k1] - 1) + i1

            if BC.bc_stokes_fix[m1]:
                K_ref = K_el[ikk, ikk]
                for jkk in range(mV * ndofV):
                    f_el[jkk] -= K_el[jkk, ikk] * BC.bc_stokes_val[m1]
                    K_el[ikk, jkk] = 0.0
                    K_el[jkk, ikk] = 0.0
                K_el[ikk, ikk] = K_ref
                f_el[ikk] = K_ref * BC.bc_stokes_val[m1]
                h_el[:] -= G_el[ikk, :] * BC.bc_stokes_val[m1]
                G_el[ikk, :] = 0.0

    # --- Modify G_el columns for fixed pressure DOFs ---
    for k2 in range(mP):
        m2 = el_conP[k2] - 1
        if BC.bc_pressured_fix[m2]:
            for k1 in range(mV):
                for i1 in range(ndofV):
                    ikk = ndofV * k1 + i1
                    G_el[ikk, k2] = 0.0

    # --- Assemble global matrix and RHS ---
    for k1 in range(mV):
        for i1 in range(ndofV):
            ikk = ndofV * k1 + i1
            m1  = ndofV * (el_con[k1] - 1) + i1

            for k2 in range(mV):
                for i2 in range(ndofV):
                    jkk = ndofV * k2 + i2
                    m2  = ndofV * (el_con[k2] - 1) + i2

                    row[idx] = m1
                    col[idx] = m2
                    val[idx] = K_el[ikk, jkk]
                    idx += 1

            for k2 in range(mP):
                jkk = k2
                m2  = el_conP[k2] - 1

                # G and G^T contributions
                row[idx] = m1
                col[idx] = NfemV + m2
                val[idx] = G_el[ikk, jkk] * pressure_scaling
                idx += 1

                row[idx] = NfemV + m2
                col[idx] = m1
                val[idx] = G_el[ikk, jkk] * pressure_scaling
                idx += 1
            if BC.bc_pressured_fix[m2]:
                # Add Dirichlet pressure contribution to f_rhs
                f_rhs[m1] = f_rhs[m1] + f_el[ikk]

            else:
                f_rhs[m1] = f_rhs[m1] + f_el[ikk]


    # --- Assemble h_rhs (pressure RHS) ---
    for k2 in range(mP):
        m2 = el_conP[k2] - 1

        h_rhs[m2] += h_el[k2] * pressure_scaling

    return f_rhs, h_rhs, row, col, val, idx

#------------------------------------------------------------------------------------------------------------------------
@njit
def _impose_BC_stokes(el_conP,el_con,CD,BC,h_rhs,f_rhs,G_el,f_el,K_el,h_el,pressure_scaling,iel,row,col,val,idx,dir_P):
    
    Nfem = (CD.NfemV+CD.NfemP)
    NfemV=CD.NfemV
    NfemP=CD.NfemP
    ndofV=CD.ndofV
    mV   =CD.mV 
    mP    = CD.mP

    
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*(el_con[k1]-1)+i1
            if BC.bc_stokes_fix[m1]:
               K_ref=K_el[ikk,ikk] 
               for jkk in range(0,mV*ndofV):
                   f_el[jkk] = f_el[jkk] -K_el[jkk,ikk]*BC.bc_stokes_val[m1]
                   K_el[ikk,jkk]=0
                   K_el[jkk,ikk]=0
               #end for
               K_el[ikk,ikk]=K_ref
               f_el[ikk]=K_ref*BC.bc_stokes_val[m1]
               h_el[:]= h_el[:] - G_el[ikk,:]*BC.bc_stokes_val[m1]
               G_el[ikk,:]=0
            #end if
        #end for
    #end for
    # assemble matrix K_mat and right hand side rhs
    for k1 in range(0,mV):
        for i1 in range(0,ndofV):
            ikk=ndofV*k1          +i1
            m1 =ndofV*(el_con[k1]-1)+i1
            for k2 in range(0,mV):
                for i2 in range(0,ndofV):
                    jkk=ndofV*k2          +i2
                    m2 =ndofV*(el_con[k2]-1)+i2
                    #K_mat[m1,m2]+=K_el[ikk,jkk]
                    #A_sparse[m1,m2] += K_el[ikk,jkk]
                    row[idx] = m1
                    
                    col[idx] = m2
                    
                    val[idx] = K_el[ikk, jkk]
                    
                    idx = idx + 1
                #end for
            #end for
            for k2 in range(0,mP):
                jkk = k2
                m2  = el_conP[k2]-1

                row[idx] = m1 
                col[idx] = NfemV + m2
                val[idx] = G_el[ikk, jkk] * pressure_scaling
                idx += 1

                row[idx] = NfemV + m2
                col[idx] = m1
                val[idx] = G_el[ikk, jkk] * pressure_scaling
                idx = idx + 1

                #G_mat[m1,m2]+=G_el[ikk,jkk]
                #A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]*pressure_scaling
                #A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]*pressure_scaling
            #end for
            f_rhs[m1] = f_rhs[m1] + f_el[ikk]
        #end for
    #end for
    for k2 in range(0,mP):
        m2=el_conP[k2]-1

        h_rhs[m2] = h_rhs[m2]+ h_el[k2]*pressure_scaling 

    return f_rhs,h_rhs,row,col,val,idx


#---------------------------------------------------------------------------------------------------------------------------

@njit
def Volume_integrals_stokes(M,CD,Cq,quad,vel,T,PL_k,shapes,iel,b_mat,N_matp,G_el,K_el,f_el,counterq,ctrl,pdb,c_mat,it,sc):
    
    #-- 
    # - Extract relevant data from the main structures
    
    # - grid nodes
    mV   = CD.mV 
    mP    = CD.mP
    mT    = CD.mT

    # - x and y coordinates
    X = np.zeros((2, mV), dtype=np.float64)

    x   = M.x[M.el_con[iel,:]-1]
    y   = M.y[M.el_con[iel,:]-1]

    X[0,:] = x
    X[1,:] = y
    # - Solution variables
    # Velocities
    V   = np.zeros((2, mV), dtype=np.float64)

    u   = vel.u[M.el_con[iel,0:mV]-1]

    v   = vel.v[M.el_con[iel,0:mV]-1]

    V[0, :] = u
    V[1, :] = v



    # Scalars 
    Tk   = T[M.el_con[iel,0:mT]-1]
    PL_k   = PL_k[M.el_con[iel,0:mT]-1]

    # - Phases 
    """
    Phases are defined in the nodes of the grid. During the computation, 
    iF computes the phase ratio in the integration point then compute per each
    of the phases that contributes to the phase ratio the material properties
    and then do a simple weighted average of the required material properties. 
    """


    Phase = M.Phase[M.el_con[iel,:]-1]

    # - Prepare phase ratio array
    phase_ratio = np.zeros((3),dtype=np.float64)
    for kq in range (0,CD.nq):

        # compute phase ratio
        rq = Cq[kq,0]
        sq = Cq[kq,1]
        phase_ratio = compute_phase_ratio_numba(Phase, phase_ratio,rq,sq,3)            
    
        # position & weight of quad. point
        weightq = Cq[kq,2]

        #-- 
        # - Create arrays shape function
        
        N_mat = shapes.NNNV[:,kq]

        #- Shape derivatives

        dN_mat = shapes.dNVdsr[:,:,kq]

        # Compute jacobian, determinant and inverse

        jcb = np.dot(dN_mat,X.T)

        jcob = np.linalg.det(jcb)

        jcbi = np.linalg.inv(jcb)

        dNX_mat = np.dot(jcbi,dN_mat)  # 2xnode matrix with dNdx and dNdy
        

        exxq = np.dot(dNX_mat[0,:],u)
        eyyq = np.dot(dNX_mat[1,:],v)
        exyq = 0.5 * np.dot(dNX_mat[1,:],u) + 0.5 * np.dot(dNX_mat[0,:],v)
    
        # Extract scalar values 
        T_q = np.dot(shapes.NNNT[:,kq],Tk)
        PL_q = np.dot(shapes.NNNT[:,kq],PL_k)


        # construct 3x8 b_mat matrix
        for i in range(mV):
            b_mat[0, 2*i    ] = dNX_mat[0,i]
            b_mat[0, 2*i + 1] = 0.0
            b_mat[1, 2*i    ] = 0.0
            b_mat[1, 2*i + 1] = dNX_mat[1,i]
            b_mat[2, 2*i    ] = dNX_mat[1,i]
            b_mat[2, 2*i + 1] = dNX_mat[0,i]


        # compute elemental a_mat matrix
        eta_q = 0.0
        for i in range(0,3):
            if phase_ratio[i] < 0.0:
                raise ValueError("Phase ratio is negative, which is not allowed.")
            elif phase_ratio[i] > 1.0:
                if np.abs(phase_ratio[i]-1)> 1e-5:
                    raise ValueError("Phase ratio is greater than 1, which is not allowed.")
                else: 
                    phase_ratio[i] = 1.0 
            if phase_ratio[i] > 0.0:
                eta_q = eta_q + phase_ratio[i] * viscosity(exxq*sc.strain,
                                                           eyyq*sc.strain,
                                                           exyq*sc.strain,
                                                           T_q*sc.Temp,
                                                           PL_q*sc.stress,
                                                           ctrl,
                                                           pdb,
                                                           it
                                                           ,i)
        eta_q = eta_q/sc.eta 
        
        rho_q = density(ctrl.option_rho,T_q*sc.Temp,PL_q*sc.stress,it)
        rho_q = rho_q/sc.rho
        quad.eta_q[counterq] = eta_q 
        
        if ctrl.pressure_bc == 1:
            g_vec = np.array([0,-ctrl.g],dtype=np.float64)
            for i in range(0,mV):
                f_el[i*2]   = f_el[i*2]   + shapes.NNNV[i,kq]*weightq * jcob * rho_q * (g_vec[0])
                f_el[i*2+1] = f_el[i*2+1] + shapes.NNNV[i,kq]*weightq * jcob * rho_q * (g_vec[1])
            
                



        K_el = K_el + b_mat.T.dot(c_mat.dot(b_mat))*weightq*jcob*eta_q
    
        for i in range(0,mP):
            N_matp[0,i] = shapes.NNNP[i,kq]
            N_matp[1,i] = shapes.NNNP[i,kq]
            N_matp[2,i] = 0.



        G_el= G_el - b_mat.T.dot(N_matp)*weightq*jcob

        counterq = counterq + 1

    return G_el,K_el,f_el,counterq,quad 
#-----------------------------------------------------------------------------------------------------------------------

#@njit
def create_stiffiness_stokes(M:MESH,
                             V:VelocityFields,
                             scalars:ScalarFields,
                             quad: QuadratureFields,
                             shapes: ShapeFunctions,
                             CD:Computational_Data,
                             BC:bc_class,
                             ctrl:NumericalControls,
                             pdb:PhaseDataBase,
                             it:int,
                             dir_P:int,
                             sc:Scal):

    #Computational data
    Nfem   = (CD.NfemV + CD.NfemP)
    NfemV  = CD.NfemV
    NfemP  = CD.NfemP
    ndofV  = CD.ndofV
    mV     = CD.mV 
    ndofP  = CD.ndofP
    mP     = CD.mP
    mT     = CD.mT

    rhs    = np.zeros(Nfem,dtype=np.float64)         # right hand side of Ax=b
    f_rhs  = np.zeros(NfemV,dtype=np.float64)        # right hand side f 
    h_rhs  = np.zeros(NfemP,dtype=np.float64)        # right hand side h 

    
    # Elemental level 
    c_mat = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float64)


    
    # Spell out the solution 
    T = scalars.T 
    PL = scalars.p_lit

    # Extract qcoord 
    rq = CD.qcoords_r
    sq = CD.qcoords_s
    wq = CD.qweights
    Cq = np.zeros((rq.shape[0], 3), dtype=np.float64)    
    Cq[:,0] = rq
    Cq[:,1] = sq
    Cq[:,2] = wq 

    # 
    #row = np.zeros((M.nel*),dtype=np.int64)

    counterq=0
    # Create the row column and val of the sparse matrix
    row = np.zeros((M.nel*280),dtype=np.int32)

    col =np.zeros((M.nel*280),dtype=np.int32)

    val = np.zeros((M.nel*280),dtype=np.float64)
    # counter for the sparse matrix
    idx = 0 
    for iel in prange(M.nel):
        """
        Construction K_el, G_el, f_el, h_el per each element -> reinitialise 
        every loop, and then sending to loop quadrature 
        
        
        """

        f_el = np.zeros((mV * ndofV),dtype=np.float64)
        K_el = np.zeros((mV * ndofV,mV * ndofV),dtype=np.float64)
        G_el = np.zeros((mV * ndofV,mP * ndofP),dtype=np.float64)
        h_el = np.zeros((mP * ndofP),dtype=np.float64)
        b_mat  = np.zeros((3,ndofV * mV),dtype=np.float64) # gradient matrix B 
        N_mat  = np.zeros((3,ndofP * mP),dtype=np.float64) # matrix to build G_el 


        G_el,K_el,f_el,counterq,quad = Volume_integrals_stokes(M,CD,Cq,quad,V,T,PL,shapes,iel,b_mat,N_mat,G_el,K_el,f_el,counterq,ctrl,pdb,c_mat,it,sc)

        f_rhs,h_rhs,row,col,val,idx   = _impose_BC_stokes(M.el_conP[iel,:],M.el_con[iel,:],CD,BC,h_rhs,f_rhs,G_el,f_el,K_el,h_el,ctrl.pressure_scaling,iel,col,row,val,idx,dir_P)
        



    rhs[0:NfemV]=f_rhs
    rhs[NfemV:Nfem]=h_rhs


    return rhs,quad,row,col,val
#-----------------------------------------------------------------------------------------------------------------------


def preamble_create_stiffiness_stokes(M,velocities,scalars,quadrature,shapes,CD,BC,ctrl,pdb,it,Nfem,dir_P,sc):

    
    rhs,quadrature,row,col,val = create_stiffiness_stokes(M,velocities,scalars,quadrature,shapes,CD,BC,ctrl,pdb,it,dir_P,sc)

    A_sparse = coo_matrix((val, (row, col)), shape=(Nfem, Nfem))

    # 2. Remove duplicates (safe and smart)
    A_sparse.sum_duplicates()

    A_sparse.eliminate_zeros()


    # 3. Convert directly to CSR (best for solving with spsolve)
    A_sparse = A_sparse.tocsr()

    # 4. Eliminate zeros (optional, slight speedup if there are many zeros)

    if ctrl.petsc == 1: 
        # Convert to PETSc format if needed
        A_sparse = scipy_sparse_to_petsc(A_sparse)
        # Create PETSc vector for the right-hand side
        rhs = numpy_rhs_to_petsc(rhs)

    return A_sparse, rhs, quadrature
#-----------------------------------------------------------------------------------------------------------------------


def solve_stokes_system(M:MESH,
                        S:Sol,
                        CD:Computational_Data,
                        BC:bc_class,
                        ctrl:NumericalControls,
                        pdb:PhaseDataBase,
                        it:int,
                        sc:Scal):
    
    
    
    print("    :::: Stokes system ::::    _")

    start_loop = timing.time()

    Nfem  = (CD.NfemV + CD.NfemP)
    NfemV = CD.NfemV
    
    dir_P = np.where((M.x == 0.0) & (M.y == 0.0))
    dir_P = dir_P[0]

    res = 1.0 
    
    u_old = S.velocities.u.copy()
    v_old = S.velocities.v.copy()
    p_old = S.scalars.p.copy()


    while res > 1e-4:
        print("    ---- ----    ")

        start = timing.time()

        A_sparse,rhs,S.quadrature = preamble_create_stiffiness_stokes(M,S.velocities,S.scalars,S.quadrature,S.shapes,CD,BC,ctrl,pdb,it,Nfem,dir_P,sc)
        print("         Assembly in %.3f s, -> %.4f s per element" % ((timing.time() - start),(timing.time() - start)/M.nel))
        start2 = timing.time()

        if ctrl.petsc == 1:
            x_petsc = rhs.duplicate()

            ksp = PETSc.KSP().create()
            ksp.setOperators(A_sparse)
            ksp.setType('preonly')  # Direct solve (no iteration)
            pc = ksp.getPC()
            pc.setType('lu')        # LU factorization
            pc.setFactorSolverType('mumps')  # or 'superlu', 'umfpack' if available
            ksp.setFromOptions()

            ksp.solve(rhs, x_petsc)
            sol = x_petsc.getArray()
        else:
            sol = sps.linalg.spsolve(A_sparse,rhs)
        
        print("         Solution current iteration in %.3f s," % ((timing.time() - start2)))


        S.velocities.u,S.velocities.v = np.reshape(sol[0:NfemV],(M.nv,2)).T
        S.scalars.p = sol[NfemV:Nfem] * ctrl.pressure_scaling
        res_u = np.linalg.norm(S.velocities.u-u_old,2)/np.linalg.norm(S.velocities.u+u_old,2)
        res_v = np.linalg.norm(S.velocities.v-v_old,2)/np.linalg.norm(S.velocities.v+v_old,2)
        res_p = np.linalg.norm(S.scalars.p-p_old,2)/np.linalg.norm(S.scalars.p+p_old,2)
        print("         res vx = %.3e, res vz = %.3e, res_p = %.3e [n.d]," %(res_u,res_v,res_p))
    
        if it>0:
           relax = ctrl.relax
           S.velocities.u = relax * S.velocities.u + (1-relax) * u_old
           S.velocities.v = relax * S.velocities.v + (1-relax) * v_old
           S.scalars.p = relax * S.scalars.p + (1-relax) * p_old
        
        p_old[:] = S.scalars.p[:]
        u_old[:] = S.velocities.u[:]
        v_old[:] = S.velocities.v[:]

        res = np.linalg.norm([res_u,res_v,res_p])
        
        print("         (res_p**2+res_vx**2+res_vz**2)^(1/2)  %.3e [n.d]," %res)
        print("         Iteration took %.3f  s " %(timing.time() - start))
        print("             [//] Solve took %.3f s " %(timing.time() - start2))
    
    print("    [||]Total Solve time: %.3f s" %(timing.time() - start_loop))
    print("    :::: Stokes system ::::    |")

    return S,0

#-----------------------------------------------------------------------------------------------------------------------
@njit
def Volume_integrals_temperature(M:MESH,
                         Cq: np.ndarray,
                         CD: Computational_Data,
                         velocities: VelocityFields,
                         T_old: np.ndarray,
                         T_new: np.ndarray,
                         PL: np.ndarray,
                         shapes: ShapeFunctions,
                         iel:int,
                         K: np.ndarray,
                         F: np.ndarray,
                         N_mat: np.ndarray,
                         counterq:int,
                         ctrl:NumericalControls,
                         pdb: PhaseDataBase,
                         tau: float,
                         it:int,
                         it_temp:int,
                         sc:Scal):
    # --
    # Initialize the matrices and vectors for the element
    
    Kd = np.zeros((CD.mT, CD.mT), dtype=np.float64)
    Ka = np.zeros((CD.mT, CD.mT), dtype=np.float64)
    Md = np.zeros((CD.mT, CD.mT), dtype=np.float64)

    # -- 
    # Time step inverse
    
    if ctrl.dt == 0.0:
        idt = 0.0 
        theta = 1.0 
    else:
        theta = 1.0 # Fully implicit 
        idt = 1.0 / ctrl.dt

    #--
    # Extrac the relevant data from Mesh and Computational Data
    
    mT = CD.mT

    nq = CD.nq

    # Element coordinates 

    x   = M.x[M.el_con[iel,0:mT]-1]

    y   = M.y[M.el_con[iel,0:mT]-1]

    # -- 
    # X coordinate of the node

    X = np.zeros((2, mT), dtype=np.float64)

    X[0, :] = x
    X[1, :] = y
    
    # X matrix must be node x 2 
    X = np.transpose(X)  
    # --    
    # Extract the velocities 
    V   = np.zeros((2, mT), dtype=np.float64)

    u   = velocities.u[M.el_con[iel,0:mT]-1]

    v   = velocities.v[M.el_con[iel,0:mT]-1]

    V[0, :] = u
    V[1, :] = v

    # -- 
    area = velocities.area[iel]

    # --
    # Extract the temperature and pressure
    T_old   = T_old[M.el_con[iel,0:mT]-1]
    T_new   = T_new[M.el_con[iel,0:mT]-1]
    PL = PL[M.el_con[iel,0:mT]-1]

    # --
    # Compute the element size
    h =  np.sqrt(4.0 * area / np.pi)
    # --
    # Scaling factors 
    
    ksc = sc.k
    Tempsc = sc.Temp
    rhosc = sc.rho
    Cpsc  = sc.Cp

    k_max = 0.0 
    beta_max = 0.0 

    tau_R = np.zeros(np.shape(F),dtype=np.float64)
    # --
    # Volume quadrature points     
    for kq in range(0,nq):

        # - 
        # weight of quad. point
        weightq              = Cq[kq]

        # -
        # Shape function NT at the current quadrature point vector of mT entries 
        N_mat[0:CD.mT]        = shapes.NNNT[:,kq]

        # -
        # Shape function derivativies dNNNVdr and dNNNVds at the current quadrature point vector of mT entries  
        dN_mat = shapes.dNTdsr[:,:,kq]

        # Compute the jacobian and its inverse
        jcb = np.dot(dN_mat,X)  # 2x2 jacobian matrix
        # Compute the determinant and inverse of the jacobian
        jcob = np.linalg.det(jcb)  # determinant of the jacobian

        jcbi = np.linalg.inv(jcb)  # inverse of the jacobian
        
        # compute dNdx & dNdy - global derivatives of the shape functions

        dNX_mat = np.dot(jcbi,dN_mat)  # 2xnode matrix with dNdx and dNdy

        # - 
        # Compute the velocity at the quadrature point 
        vel= np.dot(N_mat,V.T) 

        V_norm    = np.sqrt(vel[1]**2+vel[1]**2)+1e-12
        
        # - 
        # compute temperature and pressure at the quadrature point

        T_qO  = np.dot(N_mat,T_old)
        
        T_qN  = np.dot(N_mat,T_new)


        PL_q = np.dot(N_mat,PL)

        k_q, heatC_q, rho_q = compute_thermal_property(T_qN*Tempsc, PL_q,ctrl,it)
        
        k_q = k_q / ksc ; heatC_q = heatC_q / Cpsc ; rho_q = rho_q / rhosc


        # - 
        # Mass matrix

        Md = np.outer(N_mat,N_mat) * rho_q * heatC_q * weightq * jcob

        # compute diffusion matrix
        Kd = Kd + np.dot(dNX_mat.T, k_q * dNX_mat)  * weightq * jcob
        
        # compute advection matrix
        Ka = Ka + np.outer(N_mat,np.dot(vel,dNX_mat))* rho_q * heatC_q * weightq * jcob


        # Compute the SUPG correction: -> Compute downstream correction of the diffusion equation
        # Following aspect: 
        #compute_SUPG_aspect():
        # -- Compute beta
        beta = rho_q * heatC_q * V_norm 

        if beta > beta_max: 
            beta_max = beta 
    
        if k_q >= k_max:
            k_max = k_q


        advA = vel.dot(dNX_mat)  # advection vector at the quadrature point
    
        grad_T = np.dot(dNX_mat,T_new)
    
        adv_T  = rho_q*heatC_q * np.dot(vel,grad_T)
    
        dif_T  = - k_q*np.dot(grad_T,grad_T)
    
        dt_term = rho_q * heatC_q * (T_qN - T_qO)*idt 
    
        r = dt_term + adv_T + dif_T

        tau_R = tau_R + beta * weightq * jcob * (-r) * advA 

        counterq = counterq + 1 

    Pe       = (h * beta_max)/(4*k_max)

    par_SUPG = (h/(4*beta_max))*(cotah(Pe)-1/Pe)
    
    tau_R    = par_SUPG * tau_R

    if V_norm == 0.0:
        tau_R = tau_R*0.0

    # -- 
    # Compute the system matrix and right hand side vector 

    K = idt * Md + (Ka + Kd) 
    if (it_temp == 0) & (theta < 1.0):
        F = (idt * np.dot(Md,T_old)) - np.dot((1-theta) * (Ka + Kd + tau*0.0 ),T_old)
    else: 
        F = idt * np.dot(Md,T_old) + tau_R
        

    return K, F ,tau,vel

#-----------------------------------------------------------------------------------------------------------------------
@njit
def create_stiffness_energy(M:MESH,
                            quad:QuadratureFields,
                            velocities:VelocityFields,
                            scalars:ScalarFields,
                            shapes:ShapeFunctions,
                            CD:Computational_Data,
                            BC:bc_class,
                            rhs:np.ndarray,
                            ctrl:NumericalControls,
                            pdb:PhaseDataBase,
                            it:int,
                            it_temp:int,
                            sc:Scal):
    
    
    # Extract a few integer
    NfemT = CD.NfemT

    mT    = CD.mT 

    ndofT = CD.ndofT


    # Create the global matrix
    rhs = np.zeros(NfemT,dtype=np.float64)  # right hand side of Ax=b

    # Extract the temperature solution variable
    T_old = scalars.T_old
    T_new = scalars.T

    PL = scalars.p_lit
    nel = M.nel

    # Extract qcoord 


    wq = CD.qweights

    counterq = 0
    row    = np.zeros((nel*36),dtype=np.int32)
    col    = np.zeros((nel*36),dtype=np.int32)
    val    = np.zeros((nel*36),dtype=np.float64)

    idx = 0 
    for iel in range (nel):

        
        N_mat = np.zeros((mT),dtype=np.float64)         # shape functions
        
        b_el  = np.zeros(mT*ndofT,dtype=np.float64)          # right hand side vector
        
        K     = np.zeros((mT*ndofT,mT*ndofT),dtype=np.float64) # elemental stiffness matrix

        tau = np.zeros((mT,mT),dtype=np.float64)
        
        K,b_el,tau,vel = Volume_integrals_temperature(M,
                                                            wq,
                                                            CD,
                                                            velocities,
                                                            T_old,
                                                            T_new,
                                                            PL,
                                                            shapes,
                                                            iel,
                                                            K,
                                                            b_el,
                                                            N_mat,
                                                            counterq,
                                                            ctrl,
                                                            pdb,
                                                            tau,
                                                            it,
                                                            it_temp,
                                                            sc)
    
        # apply boundary conditions & global assembly
        rhs,row,col,val,idx = _impose_BC_temperature(BC,M.el_con[iel,:],mT,K,b_el,rhs,row,col,val,idx,it_temp)
        


    return rhs,quad,row,col,val
#-----------------------------------------------------------------------------------------------------------------------
def create_stiffness_energy_preamble(M:MESH
                                     ,S:Sol
                                     ,CD:Computational_Data
                                     ,BC:bc_class
                                     ,rhs:np.ndarray
                                     ,ctrl:NumericalControls
                                     ,pdb:PhaseDataBase
                                     ,it:int
                                     ,it_temp:int
                                     ,scal:Scal):
        
    rhs,quad,row,col,val = create_stiffness_energy(M,S.quadrature,S.velocities, S.scalars ,S.shapes,CD,BC,rhs,ctrl,pdb,it,it_temp,scal)
    
    A_mat = coo_matrix((val, (row, col)), shape=(CD.NfemT, CD.NfemT))

    # 2. Remove duplicates (safe and smart)
    A_mat.sum_duplicates()
    # 4. Eliminate zeros (optional, slight speedup if there are many zeros)

    A_mat.eliminate_zeros()

    A_mat = A_mat.tocsr()

    if ctrl.petsc == 1:
        # Convert to PETSc format if needed
        A_mat = scipy_sparse_to_petsc(A_mat)
        # Create PETSc vector for the right-hand side
        rhs_petsc = numpy_rhs_to_petsc(rhs)
    

    return A_mat,rhs,rhs_petsc,quad
#-----------------------------------------------------------------------------------------------------------------------
def solve_system_energy(M,S,CD,BC,ctrl,pdb,it,scal,flag_initial_guess = False):

    print("    :::: Energy system ::::    _")
    start_total = timing.time()
    res = 1.0
    rhs = np.zeros(CD.NfemT,dtype=np.float64)  # right hand side of Ax=b
    it_temp = 0
    S.scalars.T = S.scalars.T_old.copy()  # Initialize T with the old temperature
    while res > 1e-5:
        start_it = timing.time()
        print("    ---- ----    ")

        A_mat,rhs,rhs_petsc,S.quad = create_stiffness_energy_preamble(M,S,CD,BC,rhs,ctrl,pdb,it,it_temp,scal)

        print("         Assembly in %.3f s, -> %.3e s per element" % ((timing.time() - start_it),(timing.time() - start_it)/M.nel))

        if ctrl.petsc == 1:
            x_petsc = rhs_petsc.duplicate()
            residual = rhs_petsc.duplicate()  # Create a duplicate for the residual
            ksp = PETSc.KSP().create()
            ksp.setOperators(A_mat)
            ksp.setType('preonly')  # Direct solve (no iteration)
            pc = ksp.getPC()
            pc.setType('lu')        # LU factorization
            pc.setFactorSolverType('mumps')  # or 'superlu', 'umfpack' if available
            ksp.setFromOptions()
            ksp.solve(rhs_petsc, x_petsc)
            # Compute the residual
            sol = x_petsc.getArray()

            residual = compute_residuum_energy(M,BC,CD,ctrl,pdb,S.shapes,S.velocities,S.scalars.T_old,sol,S.scalars.p_lit,it,it_temp,scal)
            L2       = np.linalg.norm(residual,2)
            print('                    Energy residuum L2 norm is %5e'%L2 )



        else:
            sol = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs)

        res = np.linalg.norm(sol - S.scalars.T,2) / np.linalg.norm(S.scalars.T+sol,2)
        print("         :::: it = %d, Residual Energy: %.3e" %(it_temp,res))
        print("         [//]solve T: %.3f s" % (timing.time() - start_it))

        #-- 
        # Update the temperature field
        if it > 0 and flag_initial_guess == False:
            S.scalars.T = ctrl.relax * sol + (1 - ctrl.relax) * S.scalars.T
        else: 
            S.scalars.T = sol
            break 

        print("          MAX T, min T = [%.2f,%2f] degC"%(np.max(S.scalars.T*scal.Temp-273.15),np.min(S.scalars.T*scal.Temp-273.15)))
        print("    ---- ----    ")
        
        it_temp = it_temp + 1

    S.scalars.T_old = S.scalars.T.copy()
    print(".   [||] solve T: %.3f s" % (timing.time() - start_total))

    print("    :::: Energy system ::::    |")

    return S

#-----------------------------------------------------------------------------------------------------------------------
@njit
def _impose_BC_temperature(BC:bc_class,
                           el_con:int,
                           mT:int,
                           a_el:float,
                           b_el:float,
                           rhs:float,
                           row:int,
                           col:int,
                           val:float,
                           idx:int,
                           it_temp:int):


    for k1 in range(0,mT):
        m1=el_con[k1]-1
        if BC.bc_temp_fix[m1]:
           Aref=a_el[k1,k1]
           for k2 in range(0,mT):
               m2=el_con[k2]-1
               b_el[k2]-=a_el[k2,k1]*BC.bc_temp_val[m1]
               a_el[k1,k2]=0
               a_el[k2,k1]=0
           a_el[k1,k1]=Aref
           b_el[k1]=Aref*BC.bc_temp_val[m1]
        # end for
        # assemble matrix A_mat and right hand side rhs
    for k1 in range(0,mT):
        m1=el_con[k1]-1
        for k2 in range(0,mT):
            m2=el_con[k2]-1
            #A_mat[m1,m2]+=a_el[k1,k2]
            row[idx] = m1
            col[idx] = m2
            val[idx] = a_el[k1,k2]
            idx = idx + 1

        # We update rhs vector only during the first iteration in which Tn+1 = Told -> remains constant
         
        rhs[m1] = rhs[m1] + b_el[k1]

    return rhs,row,col,val,idx

#-----------------------------------------------------------------------------------------------------------------------
@njit
def compute_residuum_energy(M,BC,CD,ctrl,pdb,shapes,velocities,T_old,T_new,PL,it,it_temp,sc):
    
    
    counterq = 0 
    wq       = CD.qweights 
    mT       = CD.mT
    ndofT    = CD.ndofT
    res      = np.zeros_like(T_new)

    for iel in range (M.nel):
        
        nodes = M.el_con[iel,0:mT]-1 
        
        N_mat = np.zeros((mT),dtype=np.float64)         # shape functions
        
        b_el  = np.zeros(mT*ndofT,dtype=np.float64)          # right hand side vector
        
        K     = np.zeros((mT*ndofT,mT*ndofT),dtype=np.float64) # elemental stiffness matrix

        tau = np.zeros((mT,mT),dtype=np.float64)
        
        K,b_el,tau,vel = Volume_integrals_temperature(M,
                                                            wq,
                                                            CD,
                                                            velocities,
                                                            T_old,
                                                            T_new,
                                                            PL,
                                                            shapes,
                                                            iel,
                                                            K,
                                                            b_el,
                                                            N_mat,
                                                            counterq,
                                                            ctrl,
                                                            pdb,
                                                            tau,
                                                            it,
                                                            it_temp,
                                                            sc)
        

        res[nodes] =res[nodes] + (np.dot(K,T_new[nodes]) - b_el) 
        
    res[BC.bc_temp_fix] = T_new[BC.bc_temp_fix]-BC.bc_temp_val[BC.bc_temp_fix]

    return res





#------------------------------------------------------------------------------------------------------------------------
def compute_residuum_stokes(M,CD,sh,ctrl,):

    pass 

#------------------------------------------------------------------------------------------------------------------------
@njit 
def compute_area_elements(M:MESH,vel,scal,sh,CD:Computational_Data):
    """
    function that compute the area of each element within the domain
    and check whether or not there are problems with the mesh. 
        
    """



    for iel in range(0,M.nel):
        x   = M.x[M.el_con[iel,0:CD.mV]-1]
        y   = M.y[M.el_con[iel,0:CD.mV]-1]
        # -- 
        # X coordinate of the node
        X = np.zeros((2, CD.mV), dtype=np.float64)
        X[0,:] = x
        X[1,:] = y
        X = X.T 

        for kq in range (0,CD.nq):
            # -
            # Shape function NT at the current quadrature point vector of mT entries 
            N_mat        = sh.NNNV[:,kq]

            # -
            # Shape function derivativies dNNNVdr and dNNNVds at the current quadrature point vector of mT entries  
            dN_mat = sh.dNVdsr[:,:,kq]

            # Compurhste the jacobian and its inverse
            jcb = np.dot(dN_mat,X)  # 2x2 jacobian matrix
            # Compute the determinant and inverse of the jacobian
            jcob = np.linalg.det(jcb)  # determinant of the jacobian

            vel.area[iel]+=jcob*CD.qweights[kq]

    for iel in range(0,M.nel):

        x   = M.x[M.el_con[iel,0:CD.mT]-1]
        y   = M.y[M.el_con[iel,0:CD.mT]-1]
        # -- 
        # X coordinate of the node
        X = np.zeros((2, CD.mT), dtype=np.float64)
        X[0,:] = x
        X[1,:] = y
        X = X.T 

        for kq in range (0,CD.nq):
            # -
            # Shape function NT at the current quadrature point vector of mT entries 
            N_mat        = sh.NNNT[:,kq]

            # -
            # Shape function derivativies dNNNVdr and dNNNVds at the current quadrature point vector of mT entries  
            dN_mat = sh.dNTdsr[:,:,kq]

            # Compute the jacobian and its inverse
            jcb = np.dot(dN_mat,X)  # 2x2 jacobian matrix
            # Compute the determinant and inverse of the jacobian
            jcob = np.linalg.det(jcb)  # determinant of the jacobian

            scal.area[iel]+=jcob*CD.qweights[kq]
    
    return vel,scal
    


