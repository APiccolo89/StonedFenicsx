import sys, os
sys.path.insert(0, "/Users/wlnw570/Work/Leeds/Fenics_tutorial/Stonedphoenix")
import gmsh 
import meshio
from mpi4py                          import MPI
from petsc4py                        import PETSc
from dolfinx                         import mesh, fem, io, nls, log
from dolfinx.fem.petsc               import NonlinearProblem
from dolfinx.nls.petsc               import NewtonSolver
from dolfinx.io                      import XDMFFile, gmshio
from ufl                             import exp, conditional, eq, as_ufl
from src.create_mesh                 import Mesh
from src.numerical_control           import NumericalControls, ctrl_LHS
from src.numerical_control           import IOControls 
from src.solution                    import Solution 
from src.compute_material_property   import density_FX
from src.compute_material_property   import heat_conductivity_FX
from src.compute_material_property   import heat_capacity_FX

import ufl

from   src.scal                      import Scal
from   src.create_mesh               import Mesh
from   src.phase_db                  import PhaseDataBase
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

import matplotlib.pyplot             as plt
import compute_material_property     as cmp 
import src.scal                      as sc_f 
import basix.ufl
import time                          as timing

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from src.utils import timing_function, print_ph, interpolate_from_sub_to_main
from ufl import FacetNormal, ds, dot, sqrt, as_vector,inner, outer, grad, Identity, CellDiameter
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from dolfinx.fem import (Constant, Function, FunctionSpace, dirichletbc,
                         extract_function_spaces, form,
                         locate_dofs_topological, locate_dofs_geometrical)
from dolfinx import default_real_type, la





def dof_coords_on_facet(V, facet_tags, marker, component=None):
    """
    Return the coordinates of DOFs of V (or V.sub(component)) that lie on facets with a given marker.
    """
    mesh = V.mesh
    tdim  = mesh.topology.dim
    fdim  = tdim - 1
    gdim  = mesh.geometry.dim

    # Ensure needed connectivities exist
    mesh.topology.create_connectivity(fdim, tdim)
    mesh.topology.create_connectivity(tdim, fdim)

    # Facet indices with the given tag
    facets = facet_tags.find(marker)

    # Parent-space DOF coordinates
    X_all = V.tabulate_dof_coordinates().reshape(-1, gdim)

    if component is None:
        # scalar space: locate directly on V
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        coords = X_all[dofs]
    else:
        # vector/mixed space: locate on subspace, then map to parent DOFs
        Vs = V.sub(component)
        dofs_s = fem.locate_dofs_topological(Vs, fdim, facets)
        parent_map = Vs.dofmap.list.array  # subspace -> parent DOF ids
        coords = X_all[parent_map[dofs_s]]

    # Often multiple DOFs share the same point on higher-order spaces; deduplicate if desired
    coords = np.unique(coords, axis=0)
    return coords

@timing_function    
def block_direct_solver(a, a_p, L, bcs, V, Q, mesh, sc):
    """Solve the Stokes problem using blocked matrices and a direct
    solver."""

    # Assembler the block operator and RHS vector
    A, _, b = block_operators(a, a_p, L, bcs, V, Q)

    # Create a solver
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")

    # Set the solver type to MUMPS (LU solver) and configure MUMPS to
    # handle pressure nullspace
    pc = ksp.getPC()
    pc.setType("lu")
    use_superlu = PETSc.IntType == np.int64
    if PETSc.Sys().hasExternalPackage("mumps") and not use_superlu:
        pc.setFactorSolverType("mumps")
        pc.setFactorSetUpSolverType()
        pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
        pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)
    else:
        pc.setFactorSolverType("superlu_dist")

    # Create a block vector (x) to store the full solution, and solve
    x = A.createVecLeft()
    ksp.solve(b, x)

    # Create Functions and scatter x solution
    u, p = Function(V), Function(Q)
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    u.x.array[:offset] = x.array_r[:offset]
    p.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]

    # Compute the $L^2$ norms of the u and p vectors
    norm_u, norm_p = la.norm(u.x), la.norm(p.x)


    return u,p

#---------------------------------------------------------------------------
@timing_function    
def nested_iterative_solver():
    """Solve the Stokes problem using nest matrices and an iterative solver."""

    # Assemble nested matrix operators
    A = fem.petsc.assemble_matrix_nest(a, bcs=bcs)
    A.assemble()

    # Create a nested matrix P to use as the preconditioner. The
    # top-left block of P is shared with the top-left block of A. The
    # bottom-right diagonal entry is assembled from the form a_p11:
    P11 = fem.petsc.assemble_matrix(a_p11, [])
    P = PETSc.Mat().createNest([[A.getNestSubMatrix(0, 0), None], [None, P11]])
    P.assemble()

    # Assemble right-hand side vector
    b = fem.petsc.assemble_vector_nest(L)

    # Modify ('lift') the RHS for Dirichlet boundary conditions
    fem.petsc.apply_lifting_nest(b, a, bcs=bcs)

    # Sum contributions for vector entries that are share across
    # parallel processes
    for b_sub in b.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Set Dirichlet boundary condition values in the RHS vector
    bcs0 = fem.bcs_by_block(extract_function_spaces(L), bcs)
    fem.petsc.set_bc_nest(b, bcs0)

    # The pressure field is determined only up to a constant. We supply
    # a vector that spans the nullspace to the solver, and any component
    # of the solution in this direction will be eliminated during the
    # solution process.
    null_vec = fem.petsc.create_vector_nest(L)

    # Set velocity part to zero and the pressure part to a non-zero
    # constant
    null_vecs = null_vec.getNestSubVecs()
    null_vecs[0].set(0.0), null_vecs[1].set(1.0)

    # Normalize the vector that spans the nullspace, create a nullspace
    # object, and attach it to the matrix
    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    #assert nsp.test(A)
    A.setNullSpace(nsp)

    # Create a MINRES Krylov solver and a block-diagonal preconditioner
    # using PETSc's additive fieldsplit preconditioner
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A, P)
    ksp.setType("minres")
    ksp.setTolerances(rtol=1e-10)
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    # Define the matrix blocks in the preconditioner with the velocity
    # and pressure matrix index sets
    nested_IS = P.getNestISs()
    ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]))

    # Set the preconditioners for each block. For the top-left
    # Laplace-type operator we use algebraic multigrid. For the
    # lower-right block we use a Jacobi preconditioner. By default, GAMG
    # will infer the correct near-nullspace from the matrix block size.
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("gamg")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("jacobi")

    # Create finite element {py:class}`Function <dolfinx.fem.Function>`s
    # for the velocity (on the space `V`) and for the pressure (on the
    # space `Q`). The vectors for `u` and `p` are combined to form a
    # nested vector and the system is solved.
    u, p = Function(V), Function(Q)
    x = PETSc.Vec().createNest([la.create_petsc_vector_wrap(u.x),
                                la.create_petsc_vector_wrap(p.x)])
    ksp.solve(b, x)

    # Save solution to file in XDMF format for visualization, e.g. with
    # ParaView. Before writing to file, ghost values are updated using
    # `scatter_forward`.
    with XDMFFile(MPI.COMM_WORLD, "velocity.xdmf", "w") as ufile_xdmf:
        u.x.scatter_forward()
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(u)

    with XDMFFile(MPI.COMM_WORLD, "pressure.xdmf", "w") as pfile_xdmf:
        p.x.scatter_forward()
        pfile_xdmf.write_mesh(msh)
        pfile_xdmf.write_function(p)

    # Compute norms of the solution vectors
    norm_u = u.x.norm()
    norm_p = p.x.norm()


    return norm_u, norm_p

def block_operators(a, a_p, L, bcs, V, Q):
    """Return block operators and block RHS vector for the Stokes
    problem"""
    A = assemble_matrix_block(a, bcs=bcs); A.assemble()
    P = assemble_matrix_block(a_p, bcs=bcs); P.assemble()
    b = assemble_vector_block(L, a, bcs=bcs)

    # nullspace vector [0_u; 1_p] locally
    null_vec = A.createVecLeft()
    nloc_u = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    nloc_p = Q.dofmap.index_map.size_local
    null_vec.array[:nloc_u]  = 0.0
    null_vec.array[nloc_u:nloc_u+nloc_p] = 1.0
    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    A.setNullSpace(nsp)

    return A, P, b

def block_iterative_solver(a, a_p, L, bcs, V, Q, msh, sc):
    """Solve the Stokes problem using blocked matrices and an iterative
    solver."""

    # Assembler the operators and RHS vector
    A, P, b = block_operators(a, a_p, L, bcs, V, Q)

    # Build PETSc index sets for each field (global dof indices for each
    # field)
    V_map = V.dofmap.index_map
    Q_map = Q.dofmap.index_map
    bs_u  = V.dofmap.index_map_bs  # = mesh.dim for vector CG
    nloc_u = V_map.size_local * bs_u
    nloc_p = Q_map.size_local

    # local starts in the *assembled block vector*
    offset_u = 0
    offset_p = nloc_u

    is_u = PETSc.IS().createStride(nloc_u, offset_u, 1, comm=PETSc.COMM_SELF)
    is_p = PETSc.IS().createStride(nloc_p, offset_p, 1, comm=PETSc.COMM_SELF)


    # Create a MINRES Krylov solver and a block-diagonal preconditioner
    # using PETSc's additive fieldsplit preconditioner
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A, P)
    ksp.setTolerances(rtol=1e-10)
    ksp.setType("minres")
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
    ksp.getPC().setFieldSplitIS(("u", is_u), ("p", is_p))

    # Configure velocity and pressure sub-solvers
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("gamg")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("jacobi")

    # The matrix A combined the vector velocity and scalar pressure
    # parts, hence has a block size of 1. Unlike the MatNest case, GAMG
    # cannot infer the correct near-nullspace from the matrix block
    # size. Therefore, we set block size on the top-left block of the
    # preconditioner so that GAMG can infer the appropriate near
    # nullspace.
    ksp.getPC().setUp()
    Pu, _ = ksp_u.getPC().getOperators()
    Pu.setBlockSize(msh.topology.dim)

    # Create a block vector (x) to store the full solution and solve
    x = A.createVecRight()
    ksp.solve(b, x)

    xu = x.getSubVector(is_u)
    xp = x.getSubVector(is_p)
    u, p = fem.Function(V), fem.Function(Q)
    
    u.x.array[:] = xu.array_r
    p.x.array[:] = xp.array_r
    
    u.name, p.name = "Velocity", "Pressure"
    

        


    return  u, p


def tau(eta, u):
    return 2 * eta * ufl.sym(ufl.grad(u))

def compute_nitsche_FS(mesh, eta, tV, TV, tP, TP, dS, a1, a2, a3, gamma=100.0):
    """
    Compute the Nitsche free slip boundary condition for the slab problem.
    This is a placeholder function and should be implemented with the actual Nitsche method.
    """
    n = ufl.FacetNormal(mesh)
    h = ufl.CellDiameter(mesh)

    a1 += (
        - ufl.inner(tau(eta, tV), ufl.outer(ufl.dot(TV, n) * n, n)) * dS
        - ufl.inner(ufl.outer(ufl.dot(tV, n) * n, n), tau(eta, TV)) * dS
        + (2 * eta * gamma / h)
        * ufl.inner(ufl.outer(ufl.dot(tV, n) * n, n), ufl.outer(TV, n))
        * dS
    )
    a2 += ufl.inner(tP, ufl.dot(TV, n)) * dS
    a3 += ufl.inner(TP, ufl.dot(tV, n)) * dS

    return a1, a2, a3
#---------------------------------------------------------------------------
def set_slab_dirichlecht(ctrl, V, D, theta,sc):
    mesh = V.mesh
    tdim = mesh.topology.dim
    fdim = tdim - 1

    # facet ids
    inflow_facets  = D.facets.find(D.bc_dict['inflow'])
    outflow_facets = D.facets.find(D.bc_dict['outflow'])
    slab_facets    = D.facets.find(D.bc_dict['top_subduction'])
    X              = V.tabulate_dof_coordinates()
    dofs        = fem.locate_dofs_topological(V, fdim, slab_facets)
    
    # dofs on subspaces (entities = facet indices!)
    dofs_in_x  = fem.locate_dofs_topological(V.sub(0), fdim, inflow_facets)
    dofs_in_y  = fem.locate_dofs_topological(V.sub(1), fdim, inflow_facets)
    dofs_out_x = fem.locate_dofs_topological(V.sub(0), fdim, outflow_facets)
    dofs_out_y = fem.locate_dofs_topological(V.sub(1), fdim, outflow_facets)
    dofs_s_x   = fem.locate_dofs_topological(V.sub(0), fdim, slab_facets)
    dofs_s_y   = fem.locate_dofs_topological(V.sub(1), fdim, slab_facets)
    
    
    # scalar BCs on subspaces
    #bc_left_x   = fem.dirichletbc(PETSc.ScalarType(ctrl.v_s[0]), dofs_in_x,  V.sub(0))
    #bc_left_y   = fem.dirichletbc(PETSc.ScalarType(ctrl.v_s[1]), dofs_in_y,  V.sub(1))
    #bc_bottom_x = fem.dirichletbc(PETSc.ScalarType(ctrl.v_s[0]*np.cos(theta)), dofs_out_x, V.sub(0))
    #bc_bottom_y = fem.dirichletbc(PETSc.ScalarType(ctrl.v_s[0]*np.sin(theta)), dofs_out_y, V.sub(1))
    
    #nx,ny = compute_normal(X,dofs)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=D.facets)
  
    n = ufl.FacetNormal(mesh)                    # exact facet normal in weak form
    v_slab = float(np.linalg.norm(ctrl.v_s))  # slab velocity magnitude

    v_const = ufl.as_vector((1.0, 0.0))
    
    proj = ufl.Identity(mesh.geometry.dim) - ufl.outer(n, n)  # projector onto the tangential plane
    t = ufl.dot(proj, v_const)                     # tangential velocity vector on slab
    t_hat = t / ufl.sqrt(ufl.inner(t, t))    
    v_project = v_slab * t_hat  # projected tangential velocity vector on slab
    
    w = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(w, v) * ds(D.bc_dict['top_subduction'])        # boundary mass matrix (vector)
    L = ufl.inner(v_project, v) * ds(D.bc_dict['top_subduction'])

    ubc = fem.petsc.LinearProblem(
        a, L,
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1e-12,
        }
    ).solve()  # ut_h \in V

    # DOFs of each component on the slab (in the parent V index space)
    dofs_s_x = fem.locate_dofs_topological(V.sub(0), fdim, slab_facets)
    dofs_s_y = fem.locate_dofs_topological(V.sub(1), fdim, slab_facets)

    bcx = fem.dirichletbc(ubc.sub(0), dofs_s_x)
    bcy = fem.dirichletbc(ubc.sub(1), dofs_s_y)
    return [bcx, bcy]
#---------------------------------------------------------------------------
def set_wedge_dirichlecht(ctrl, V, D, theta,sc):
    mesh = V.mesh
    tdim = mesh.topology.dim
    fdim = tdim - 1

    # facet ids
    slab_facets    = D.facets.find(D.bc_dict['slab'])
    X              = V.tabulate_dof_coordinates()
    dofs        = fem.locate_dofs_topological(V, fdim, slab_facets)
    
    # dofs on subspaces (entities = facet indices!)
    
    #nx,ny = compute_normal(X,dofs)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=D.facets)
  
    n = ufl.FacetNormal(mesh)                    # exact facet normal in weak form
    v_slab = float(np.linalg.norm(ctrl.v_s))  # slab velocity magnitude

    v_const = ufl.as_vector((1.0, 0.0))
    
    proj = ufl.Identity(mesh.geometry.dim) - ufl.outer(n, n)  # projector onto the tangential plane
    t = ufl.dot(proj, v_const)                     # tangential velocity vector on slab
    t_hat = t / ufl.sqrt(ufl.inner(t, t))    
    v_project = v_slab * t_hat  # projected tangential velocity vector on slab
    
    w = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(w, v) * ds(D.bc_dict['slab'])        # boundary mass matrix (vector)
    L = ufl.inner(v_project, v) * ds(D.bc_dict['slab'])

    ubc = fem.petsc.LinearProblem(
        a, L,
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1e-12,
        }
    ).solve()  # ut_h \in V

    # DOFs of each component on the slab (in the parent V index space)
    dofs_s_x = fem.locate_dofs_topological(V.sub(0), fdim, slab_facets)
    dofs_s_y = fem.locate_dofs_topological(V.sub(1), fdim, slab_facets)

    bcx = fem.dirichletbc(ubc.sub(0), dofs_s_x)
    bcy = fem.dirichletbc(ubc.sub(1), dofs_s_y)
    noslip = np.zeros(mesh.geometry.dim, dtype=PETSc.ScalarType)
    dofs_over = fem.locate_dofs_topological(V, fdim, D.facets.find(D.bc_dict['overriding']))
    bc_plate = fem.dirichletbc(noslip, dofs_over, V)
    
    return [bcx, bcy,bc_plate]
#---------------------------------------------------------------------------
@timing_function    
def set_Stokes_Wedge(pdb,sc,M,ctrl):
    """
    Input:
    PL  : lithostatic pressure field
    T_on: temperature field
    pdb : phase data base
    sc  : scaling
    g   : gravity vector
    M   : Mesh object
    Output:
    F   : Stokes problem for the slab/Solution -> still to decide
    ---
    Facet marker = 1 
    T
    
    
    The slab proble is by definition linear. First, we are speaking of an object that moves at constant velocity from top to bottom, by definition, it is not deforming. Secondly
    why introducing non-linear rheology? Would be a waste of time 
    Considering the slab problem linear implies that can be computed only once or potentially whenever the slab velocity changes. For example, if the age of the incoming slab changes, the problem is still linear as the temperature is moving as a function of the velocity field. 
    Then depends wheter or not the velocity field is constantly changing or stepwise changing.
    ---
    Following Nate Sime's approach, I will use a Nitsche method to impose the free slip boundary condition on the slab.
    """
    
    from ufl import inner, grad, div, Identity, dx, Measure, dot, sym, Constant
    
    print_ph("[] - - - -> Solving the slab's stokes problem <- - - - []")
    
    mesh = M.domainB.smesh
    #-----
    # Extract the relevant information from the mesh object
    V_subs0 = M.domainB.solSTK.sub(0)
    p_subs0 = M.domainB.solSTK.sub(1)
    V_subs, _ = V_subs0.collapse()
    p_subs, _ = p_subs0.collapse()
    
    tV = ufl.TrialFunction(V_subs)
    TV = ufl.TestFunction(V_subs)
    tP = ufl.TrialFunction(p_subs)
    TP = ufl.TestFunction(p_subs)
    
    eta_weg = fem.Constant(mesh, PETSc.ScalarType(float(pdb.eta[2])))
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=M.domainB.facets)
    
    a_1 = ufl.inner(2*eta_weg*ufl.sym(ufl.grad(tV)), ufl.grad(TV)) * ufl.dx
    a_2 = - ufl.inner(ufl.div(TV), tP) * ufl.dx
    a_3 = - ufl.inner(TP, ufl.div(tV)) * ufl.dx
    dS_over =  ds(M.domainB.bc_dict["overriding"])
    
    f  = fem.Constant(mesh, [0.0]*mesh.geometry.dim)
    f2 = fem.Constant(mesh, 0.0)
    L  = [ufl.inner(f, TV)*ufl.dx, ufl.inner(f2, TP)*ufl.dx]
    #a_1, a_2, a_3 = compute_nitsche_NS(mesh, eta_weg, tV, TV, tP, TP, dS_bot, a_1, a_2, a_3, 50.0)

    #a_1, a_2, a_3 = compute_nitsche_FS(mesh, eta_slab, tV, TV, tP, TP, dS_top, a_1, a_2, a_3,100.0)
    
    a   = fem.form([[a_1, a_2],[a_3, None]])
    a_p0 = fem.form(ufl.inner(TP, tP) * ufl.dx)
    a_p  = fem.form([[a_1, None],[None, a_p0]])   # make this a fem.form too
    L = fem.form(L)
    # -> Inflow and outflow boundary condition
    bcs = set_wedge_dirichlecht(ctrl,V_subs,M.domainB,M.g_input.theta_out_slab,sc)
    #bcs = []
    u, p = block_direct_solver(a, a_p, L, bcs, V_subs, p_subs, mesh, sc)

    
    if MPI.COMM_WORLD.rank == 0:
        pt_save = 'output/'
        if not os.path.exists(pt_save):
            os.makedirs(pt_save)
        pt_save = os.path.join(pt_save, "wedge")
        if not os.path.exists(pt_save):
            os.makedirs(pt_save)    
    
    
    with XDMFFile(MPI.COMM_WORLD, "%s/velocity.xdmf"%pt_save, "w") as ufile_xdmf:
        
        element = basix.ufl.element("Lagrange", "triangle", 1, shape=(mesh.geometry.dim,))
        u_triangular = fem.functionspace(mesh,element)
        u_T = fem.Function(u_triangular)
        u_T.name = "Velocity"
        u_T.interpolate(u)
        u_T.x.array[:] = u_T.x.array[:]*(sc.L/sc.T)/sc.scale_vel
        u_T.x.scatter_forward()

        ufile_xdmf.write_mesh(mesh)
        ufile_xdmf.write_function(u_T)

    with XDMFFile(MPI.COMM_WORLD, "%s/pressure.xdmf"%pt_save, "w") as pfile_xdmf:
        p.x.scatter_forward()

        p2 = fem.Function(p_subs)
        p2.name = "Pressure"
        p2.interpolate(p)
        p2.x.array[:] = p2.x.array[:]*sc.stress 
        p2.x.scatter_forward()
        pfile_xdmf.write_mesh(mesh)
        pfile_xdmf.write_function(p2)
    
    
    return u




#----------------------------------------------------------------------------
@timing_function    
def set_Stokes_Slab(pdb,sc,M,ctrl):
    """
    Input:
    PL  : lithostatic pressure field
    T_on: temperature field
    pdb : phase data base
    sc  : scaling
    g   : gravity vector
    M   : Mesh object
    Output:
    F   : Stokes problem for the slab/Solution -> still to decide
    ---
    Facet marker = 1 
    T
    
    
    The slab proble is by definition linear. First, we are speaking of an object that moves at constant velocity from top to bottom, by definition, it is not deforming. Secondly
    why introducing non-linear rheology? Would be a waste of time 
    Considering the slab problem linear implies that can be computed only once or potentially whenever the slab velocity changes. For example, if the age of the incoming slab changes, the problem is still linear as the temperature is moving as a function of the velocity field. 
    Then depends wheter or not the velocity field is constantly changing or stepwise changing.
    ---
    Following Nate Sime's approach, I will use a Nitsche method to impose the free slip boundary condition on the slab.
    """
    
    from ufl import inner, grad, div, Identity, dx, Measure, dot, sym, Constant
    
    print_ph("[] - - - -> Solving the slab's stokes problem <- - - - []")
    
    mesh = M.domainA.smesh
    #-----
    # Extract the relevant information from the mesh object
    V_subs0 = M.domainA.solSTK.sub(0)
    p_subs0 = M.domainA.solSTK.sub(1)
    V_subs, _ = V_subs0.collapse()
    p_subs, _ = p_subs0.collapse()
    
    tV = ufl.TrialFunction(V_subs)
    TV = ufl.TestFunction(V_subs)
    tP = ufl.TrialFunction(p_subs)
    TP = ufl.TestFunction(p_subs)
    
    eta_slab = fem.Constant(mesh, PETSc.ScalarType(float(pdb.eta[0])))
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=M.domainA.facets)
    
    a_1 = ufl.inner(2*eta_slab*ufl.sym(ufl.grad(tV)), ufl.grad(TV)) * ufl.dx
    a_2 = - ufl.inner(ufl.div(TV), tP) * ufl.dx
    a_3 = - ufl.inner(TP, ufl.div(tV)) * ufl.dx
    
    dS_top = ds(M.domainA.bc_dict["top_subduction"])
    dS_bot = ds(M.domainA.bc_dict["bot_subduction"])
    
    f  = fem.Constant(mesh, [0.0]*mesh.geometry.dim)
    f2 = fem.Constant(mesh, 0.0)
    L  = [ufl.inner(f, TV)*ufl.dx, ufl.inner(f2, TP)*ufl.dx]
    #a_1, a_2, a_3, L = compute_nitsche_moving_wall(mesh, eta_slab, tV, TV, tP, TP, dS_top, a_1, a_2, a_3, L ,100,ctrl.v_s[0])
    a_1, a_2, a_3 = compute_nitsche_FS(mesh, eta_slab, tV, TV, tP, TP, dS_bot, a_1, a_2, a_3, 50.0)

    #a_1, a_2, a_3 = compute_nitsche_FS(mesh, eta_slab, tV, TV, tP, TP, dS_top, a_1, a_2, a_3,100.0)
    
    a   = fem.form([[a_1, a_2],[a_3, None]])
    a_p0 = fem.form(ufl.inner(TP, tP) * ufl.dx)
    a_p  = fem.form([[a_1, None],[None, a_p0]])   # make this a fem.form too
    L = fem.form(L)
    # -> Inflow and outflow boundary condition
    bcs = set_slab_dirichlecht(ctrl,V_subs,M.domainA,M.g_input.theta_out_slab,sc)
    #bcs = []
    u, p = block_direct_solver(a, a_p, L, bcs, V_subs, p_subs, mesh, sc)

    #print("Stokes slab problem solved")
    F1 = compute_boundary_flux(u,M,'top_subduction')
    F2 = compute_boundary_flux(u,M,'bot_subduction')
    F3 = compute_boundary_flux(u,M,'inflow')
    F4 = compute_boundary_flux(u,M,'outflow')
    div_form = ufl.div(u) * ufl.dx
    div_int = fem.assemble_scalar(fem.form(div_form))

    total_flux = F1 + F2 + F3 + F4
    incoming_flux = ctrl.v_s[0] * 130e3/sc.L 
    print_ph("[] - - - -> Solved <- - - - []")
    print_ph(f"// - - - /Relative Total flux on the slab boundaries    : {total_flux/incoming_flux:.2e}[]/")
    print_ph(f"// - - - /Relative Total divergence integral            : {div_int/incoming_flux:.2e}[]/")
    print_ph(f"// - - - /Relative flux across the top slab abs.   [MW] : {np.abs(F1)/incoming_flux:.2e}[]/")
    print_ph(f"// - - - /Relative flux accorss bottom slab abs.   [FS] : {np.abs(F2)/incoming_flux:.2e}[]/")
    print_ph(f"// - - - /Relative influx                   abs.   [DN] : {np.abs(F3)/incoming_flux:.5e}[]/")
    print_ph(f"// - - - /Relative outflux                  abs.   [DN] : {np.abs(F4)/incoming_flux:.5e}[]/")
    print_ph(f"// - - - /Flux error                        abs.   [DN] : {(np.abs(F4)-np.abs(F3))/(np.abs(F4)+np.abs(F3)):.5e}[]/")
    print_ph("[============================]")
    print_ph("")
    print_ph("               _")
    print_ph("               :")
    print_ph("[] - - - -> Finished <- - - - []")
    print_ph("               :")
    print_ph("               _")
    
    if MPI.COMM_WORLD.rank == 0:
        pt_save = 'output/'
        if not os.path.exists(pt_save):
            os.makedirs(pt_save)
        pt_save = os.path.join(pt_save, "slab")
        if not os.path.exists(pt_save):
            os.makedirs(pt_save)    
    
    
    with XDMFFile(MPI.COMM_WORLD, "%s/velocity.xdmf"%pt_save, "w") as ufile_xdmf:
        
        element = basix.ufl.element("Lagrange", "triangle", 1, shape=(mesh.geometry.dim,))
        u_triangular = fem.functionspace(mesh,element)
        u_T = fem.Function(u_triangular)
        u_T.name = "Velocity"
        u_T.interpolate(u)
        u_T.x.array[:] = u_T.x.array[:]*(sc.L/sc.T)/sc.scale_vel
        u_T.x.scatter_forward()

        ufile_xdmf.write_mesh(mesh)
        ufile_xdmf.write_function(u_T)

    with XDMFFile(MPI.COMM_WORLD, "%s/pressure.xdmf"%pt_save, "w") as pfile_xdmf:
        p.x.scatter_forward()

        p2 = fem.Function(p_subs)
        p2.name = "Pressure"
        p2.interpolate(p)
        p2.x.array[:] = p2.x.array[:]*sc.stress 
        p2.x.scatter_forward()
        pfile_xdmf.write_mesh(mesh)
        pfile_xdmf.write_function(p2)
    
    
    return u

#---------------------------------------------------------------------------
def compute_boundary_flux(u,M,boundary_id):
    mesh = M.domainA.smesh
    facet_tags = M.domainA.facets            # MeshTags for facets (dim-1)
    tag = M.domainA.bc_dict[boundary_id]       # <- your boundary id

    # Boundary measure with tags
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    n = ufl.FacetNormal(mesh)

    # Signed flux (outward positive)
    flux_form = ufl.dot(u, n) * ds(tag)
    flux_local = fem.assemble_scalar(fem.form(flux_form))
    flux = mesh.comm.allreduce(flux_local, op=MPI.SUM)
    return flux
#---------------------------------------------------------------------------
@timing_function
def initial_temperature_field(M, ctrl, lhs):
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
        T_gr = (-M.g_input.lt_d-0)/(ctrl.Tmax-ctrl.Ttop)
        T_gr = T_gr**(-1) 
        
        bc_fun = fem.Function(X)
        bc_fun.x.array[dofs_dirichlet] = ctrl.Ttop + T_gr * cd_dof[dofs_dirichlet,1]
        bc_fun.x.scatter_forward()
    """    
    #- Create part of the thermal field: create function, extract dofs, 
    X     = M.Sol_SpaceT 
    T_i_A = fem.Function(X)
    cd_dof = X.tabulate_dof_coordinates()
    T_i_A.x.array[:] = griddata(-lhs.z, lhs.LHS, cd_dof[:,1], method='nearest')
    T_i_A.x.scatter_forward() 
    #- 
    T_gr = (-M.g_input.lt_d-0)/(ctrl.Tmax-ctrl.Ttop)
    T_gr = T_gr**(-1) 
    
    T_expr = fem.Function(X)
    ind_A = np.where(cd_dof[:,1] >= -M.g_input.lt_d)[0]
    ind_B = np.where(cd_dof[:,1] < -M.g_input.lt_d)[0]
    T_expr.x.array[ind_A] = ctrl.Ttop + T_gr * cd_dof[ind_A,1]
    T_expr.x.array[ind_B] = ctrl.Tmax
    T_expr.x.scatter_forward()
        

    expr = conditional(
        reduce(Or,[eq(M.phase, i) for i in [2, 3, 4, 5]]),
        T_expr,
        T_i_A
    )
    
    v = ufl.TestFunction(X)
    u = ufl.TrialFunction(X)
    T_i = fem.Function(X)
    a = u * v * ufl.dx 
    L = expr * v * ufl.dx
    prb = fem.petsc.LinearProblem(a,L,u=T_i)
    prb.solve()
    return T_i 

#---------------------------------------------------------------------------

@timing_function
def set_lithostatic_problem(PL, T_o, tPL, TPL, pdb, sc, g, M ):
    """
    PL  : function
    T_o : previous Temperature field
    tPL : trial function for lithostatic pressure 
    TPL : test function for lithostatic pressure
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

    
    flag = 1 
    fdim = M.mesh.topology.dim - 1    
    top_facets   = M.mesh_Ftag.find(1)
    top_dofs    = fem.locate_dofs_topological(M.Sol_SpaceT, M.mesh.topology.dim-1, top_facets)
    bc = fem.dirichletbc(0.0, top_dofs, M.Sol_SpaceT)
    
    # -> yeah only rho counts here. 
    if (np.all(pdb.option_rho) == 0):
        flag = 0
        bilinear = ufl.dot(ufl.grad(TPL), ufl.grad(tPL)) * ufl.dx
        linear   = ufl.dot(ufl.grad(TPL), density_FX(pdb, T_o, PL, M.phase,M.mesh)*g) * ufl.dx
        problem = fem.petsc.LinearProblem(bilinear, linear, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu","pc_factor_mat_solver_type": "mumps"})
        
        PL = problem.solve()
                
    
    if flag !=0: 
        # Solve lithostatic pressure - Non linear 
        F = ufl.dot(ufl.grad(PL), ufl.grad(TPL)) * ufl.dx - ufl.dot(ufl.grad(TPL), density_FX(pdb, T_o, PL, M.phase,M.mesh)*g) * ufl.dx

        problem = fem.petsc.NonlinearProblem(F, PL, bcs=[bc])
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-5
        solver.report = True
        ksp = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "gmres"
        opts[f"{option_prefix}ksp_rtol"] = 1.0e-5
        ksp.setFromOptions()
        n, converged = solver.solve(PL)
    local_max = np.max(PL.x.array)
    print_ph("[] - - - -> Solved <- - - - []")

    # Global min and max using MPI reduction
    global_max = M.comm.allreduce(local_max, op=MPI.MAX)
    print_ph(f"// - - - /Global max lithostatic pressure is    : {global_max*sc.stress/1e9:.2f}[GPa]/")
    print_ph("               _")
    print_ph("               :")
    print_ph("[] - - - -> Finished <- - - - []")
    print_ph("               :")
    print_ph("               _")
    return PL 

#---------------------------------------------------------------------------

def set_steady_state_thermal_problem(PL, T_on, tPL, TPL, pdb, sc, g, M ):
    
    
    
    adv  = ufl.inner( density_FX(pdb, T, P, M.phase,M.mesh) * heat_capacity_FX(pdb, T, M.phase,M.mesh) * vel,  ufl.grad(T)) * q * ufl.dx 
    cond = ufl.inner( heat_conductivity_FX(pdb, T, P, M.phase,M.mesh) * ufl.grad(T), ufl.grad(q)) * q * ufl.dx
    
    F = adv + cond
    
    return F

#---------------------------------------------------------------------------

def strain_rate(vel):
    
    return ufl.sym(ufl.grad(vel))
#---------------------------------------------------------------------------
    
def eps_II(vel):
 
    e = strain_rate(vel)
    
    eII = ufl.sqrt(2 * ufl.inner(e,e)) 
    
    # Portion of the model do not have any strain rate, for avoiding blow up, I just put a fictitious low strain rate
    
    
    return eII 
#---------------------------------------------------------------------------

def sigma(eta, u, p):
    
    sigma = 2 * eta * strain_rate(u) - p * ufl.Identity(2) 
    
    return sigma 


#---------------------------------------------------------------------------
def linear_stokes_solver(a, b, bcs, M):
    
    
    
    
    return u,p 

#---------------------------------------------------------------------------


def set_dirichlet_inflow():
    
    return bc


#---------------------------------------------------------------------------
@timing_function
def main_solver_steady_state(M, S, ctrl, pdb, sc, lhs ): 
    """
    To Do explanation: 
    
    
    
    -- 
    Random developing notes: The idea is to solve temperature, and lithostatic pressure for the entire domain and solve two small system for the slab [a pipe of 130 km] and the wedge. from these two small system-> interpolate the velocity into the main mesh and resolve it. 
    --- 
    A. Solve lithostatic pressure: 
    ->whole mesh 
    B. Solve Slab -> function to solve stokes problem -> class for the stokes solver -> bc -> specific function ? 
    C. Solve Wedge -> function to solve stokes problem -> class for the stokes solver -> bc 
    
    D. Merge the velocities field in only one big field -> plug in the temperature solve and solve for the whole mesh
    ----> Slab by definition with a fixed geometry is undergoing to rigid motion 
            -> No deformation -> no strain rate -> constant viscosity is good enough yahy 
            => SOLVE ONLY ONE TIME and whenever you change velocity of the slab. [change age is for free]
    ----> Wedge -> Non linear {T & ε, with/out P_l}/ Linear {T,with/out P_l}
            -> In case of temperature dependency, or pressure dependency -> each time the solution must be computed [each iteration/timestep]
    ---- Crust in any case useless and it is basically there for being evolving only thermally 
    """
    
    
    
    
    
    # Segregate solvers seems the most reasonable solution -> I can use also a switch between the options, such that I can use linear/non linear solver 
    # -> BC -> 
    
    # -- 
    # -- 
    Stk = M.Sol_SpaceSTK    
    #------
    V0    = Stk.sub(0) # Velocity space
    P0    = Stk.sub(1) # Pressure space
    V, _ = V0.collapse()  # Velocity function space
    P, _ = P0.collapse()  # Pressure function space
    #------
    u_global = fem.Function(V)  # Global velocity function
    PL          = fem.Function(M.Sol_SpaceT)
    #------
    
    
    T_o = initial_temperature_field(M, ctrl, lhs)
    # -- Test and trial function for pressure and temperature 
    tT =  ufl.TrialFunction(M.Sol_SpaceT); tPL = ufl.TrialFunction(M.Sol_SpaceT) 
    TT = ufl.TestFunction(M.Sol_SpaceT)  ; TPL = ufl.TestFunction(M.Sol_SpaceT)
    # -- 
    g = fem.Constant(M.mesh, PETSc.ScalarType([0.0, -ctrl.g]))    
    
    # Main Loop for the convergence -> check residuum of each subsystem 
    
    PL       = set_lithostatic_problem(PL, T_o, tPL, TPL, pdb, sc, g, M )
    
    u_slab   = set_Stokes_Slab(pdb,sc,M,ctrl)
    
    u_global = interpolate_from_sub_to_main(u_global, u_slab,V,M.domainA.solSTK.sub(0).collapse()[0],M.domainA.scell)
        
    u_wedge  = set_Stokes_Wedge(pdb,sc,M,ctrl)

    u_wedge = interpolate_from_sub_to_main(u_global, u_wedge,V,M.domainB.solSTK.sub(0).collapse()[0],M.domainB.scell)
    

    # interpolate the velocity field from the slab to the main mesh
    
    
    
    # -> Interpolate the velocity field from the slab and wedge to the main mesh
    
    
    # Set the temperature problem 
    #F_T = set_steady_state_thermal_problem(PL, T_on, vel, Ten ,pdb , sc, M)
    
    # Check residuum and plot the residuum plot 
    
    # Save the solution 
    
    return  

#---------------------------------------------------------------------------
def unit_test(): 
    
    
    from phase_db import PhaseDataBase
    from phase_db import _generate_phase
    from thermal_structure_ocean import compute_initial_LHS

    
    from create_mesh import unit_test_mesh
    # Create scal 
    sc = Scal(L=660e3,Temp = 1350,eta = 1e21, stress = 1e9)
    
    ioctrl = IOControls(test_name = 'Debug_test',
                        path_save = '../Debug_mesh',
                        sname = 'Experimental')
    ioctrl.generate_io()
    print_ph("[] - - - -> Creating mesh <- - - - []")
    # Create mesh 
    M = unit_test_mesh(ioctrl, sc)
            
    print_ph("[] - - - -> Creating numerical controls <- - - - []")
    ctrl = NumericalControls()
    
    ctrl = sc_f._scaling_control_parameters(ctrl, sc)
    
    print_ph("[] - - - -> Phase Database <- - - - []")

    
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
    
    print_ph("[] - - - -> Left thermal boundary condition <- - - - []")

    lhs_ctrl = ctrl_LHS()

    lhs_ctrl = sc_f._scale_parameters(lhs_ctrl, sc)
    
    lhs_ctrl = compute_initial_LHS(ctrl,lhs_ctrl, sc, pdb)

    
    # call the lithostatic pressure 
    
    S   = Solution()
    
    main_solver_steady_state(M, S, ctrl, pdb, sc, lhs_ctrl )
    
    pass 
    

if  __name__ == '__main__':
    unit_test()