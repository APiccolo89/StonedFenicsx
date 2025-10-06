# FEniCSx ≥ 0.7
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import mesh, fem, io
from dolfinx.io.gmshio import model_to_mesh
import gmsh
from petsc4py import PETSc

# ------------------------------
# 1) Mesh + facet tags (gmsh)
# ------------------------------
def build_pipe_with_cylinder(Lx=2.0, Ly=1.0, R=0.20, center=(0.5, 0.5),
                             inlet_tag=15, wall_tag=16, cyl_tag=17,
                             outlet0_tag=18, outlet1_tag=19, res=0.05):
    gmsh.initialize()
    gmsh.model.add("pipe_cyl")

    cx, cy = center
    # Rectangle and disk
    rect = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, Lx, Ly)
    disk = gmsh.model.occ.addDisk(cx, cy, 0.0, R, R)
    # Subtract the cylinder from the rectangle
    fluid, _ = gmsh.model.occ.cut([(2, rect)], [(2, disk)], removeObject=True, removeTool=False)
    gmsh.model.occ.synchronize()

    # Size field
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), res)

    # Classify boundary curves by center-of-mass
    bcurves = [c[1] for c in gmsh.model.getEntities(dim=1)]
    inlet, walls, cyl, out0, out1 = [], [], [], [], []
    for tag in bcurves:
        xcm, ycm, _ = gmsh.model.occ.getCenterOfMass(1, tag)
        # Cylinder (distance ~ R from center)
        if np.isclose(np.hypot(xcm - cx, ycm - cy), R, atol=1e-6):
            cyl.append(tag)
        # Inlet: x ~ 0
        elif np.isclose(xcm, 0.0, atol=1e-12):
            inlet.append(tag)
        # Right side (outlets): x ~ Lx; split by y half
        elif np.isclose(xcm, Lx, atol=1e-12):
            (out1 if ycm > 0.5*Ly else out0).append(tag)
        # Walls: y ~ 0 or y ~ Ly
        elif np.isclose(ycm, 0.0, atol=1e-12) or np.isclose(ycm, Ly, atol=1e-12):
            walls.append(tag)
        else:
            # Fallback: anything else goes to walls
            walls.append(tag)

    # Physical groups
    surf_tag = gmsh.model.addPhysicalGroup(2, [fluid[0][1]], tag=1)
    gmsh.model.setPhysicalName(2, surf_tag, "fluid")
    if inlet:   gmsh.model.addPhysicalGroup(1, inlet, inlet_tag)
    if walls:   gmsh.model.addPhysicalGroup(1, walls, wall_tag)
    if cyl:     gmsh.model.addPhysicalGroup(1, cyl,   cyl_tag)
    if out0:    gmsh.model.addPhysicalGroup(1, out0,  outlet0_tag)
    if out1:    gmsh.model.addPhysicalGroup(1, out1,  outlet1_tag)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)

    msh, ct, ft = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return msh, ct, ft

mesh, ct, ft = build_pipe_with_cylinder()

# ------------------------------
# 2) Spaces and helpers
# ------------------------------
V = fem.VectorFunctionSpace(mesh, ("Lagrange", 2))
Q = fem.FunctionSpace(mesh, ("Lagrange", 1))

tV = ufl.TrialFunction(V); TV = ufl.TestFunction(V)
tP = ufl.TrialFunction(Q); TP = ufl.TestFunction(Q)

x = ufl.SpatialCoordinate(mesh)
n = ufl.FacetNormal(mesh)
h = ufl.CellDiameter(mesh)

mu   = fem.Constant(mesh, PETSc.ScalarType(100.0))
p_io = fem.Constant(mesh, PETSc.ScalarType(1000.0))
f    = fem.Constant(mesh, PETSc.ScalarType((0.0, 0.0)))

# Measures by facet tags
ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
INLET, WALLS, CYL, OUT0, OUT1 = 15, 16, 17, 18, 19
ds_inlet, ds_walls = ds(INLET), ds(WALLS)
ds_cyl, ds_out0, ds_out1 = ds(CYL), ds(OUT0), ds(OUT1)

def eps(u):  # symmetric gradient
    return ufl.sym(ufl.grad(u))

# ------------------------------
# 3) Inlet & wall Dirichlet BCs
# ------------------------------
fdim = mesh.topology.dim - 1
inlet_facets  = ft.find(INLET)
wall_facets   = ft.find(WALLS)

# Vector inlet profile u_in = (sin(pi*y), 0)
u_in = fem.Function(V)
def inlet_expr(x):
    return np.vstack((np.sin(np.pi * x[1]), np.zeros_like(x[1])))
u_in.interpolate(inlet_expr)

# Zero wall
u_wall = fem.Function(V)
u_wall.x.array[:] = 0.0

dofs_inlet = fem.locate_dofs_topological(V, fdim, inlet_facets)
dofs_walls = fem.locate_dofs_topological(V, fdim, wall_facets)

bc_inlet = fem.dirichletbc(u_in, dofs_inlet, V)
bc_walls = fem.dirichletbc(u_wall, dofs_walls, V)
bcs = [bc_inlet, bc_walls]

# ------------------------------
# 4) Base Stokes bilinear forms
# ------------------------------
a11 = 2*mu*ufl.inner(eps(tV), eps(TV)) * ufl.dx     # velocity-velocity
a12 = - ufl.inner(tP, ufl.div(TV)) * ufl.dx         # pressure onto v
a21 = - ufl.inner(TP, ufl.div(tV)) * ufl.dx         # velocity onto q
a22 = 0 * ufl.inner(tP, TP) * ufl.dx                # (zero block)

L1 = ufl.inner(f, TV) * ufl.dx \
     - ufl.inner(p_io, ufl.dot(TV, n)) * ds_out1    # weak p at outlet1
L2 = 0 * TP * ufl.dx

# ------------------------------
# 5) Free-slip on cylinder
#    (choose ONE of the two variants)
# ------------------------------

# Variant A: Symmetric Nitsche (recommended)
beta = fem.Constant(mesh, PETSc.ScalarType(20.0))
tVn = ufl.dot(tV, n) * n
TVn = ufl.dot(TV, n) * n

# (i) shear parts in uu-block
a11 += ( - ufl.inner(2*mu*eps(tV)*n, TVn)          # consistency
         - ufl.inner(2*mu*eps(TV)*n, tVn)          # symmetry
         + (beta*mu/h) * ufl.inner(tVn, TVn) ) * ds_cyl

# (ii) pressure couplings on the boundary (signs are "+")
a12 += ( + ufl.inner(tP, ufl.dot(TV, n)) ) * ds_cyl
a21 += ( + ufl.inner(TP, ufl.dot(tV, n)) ) * ds_cyl

# ---- OR ----
# Variant B: Simple penalty on (u·n)=0 only (comment Variant A above, then use:)
#sigma = fem.Constant(mesh, PETSc.ScalarType(2.0))  # Babuska-type exponent
# a11 += ( - ufl.inner(2*mu*eps(tV)*n, TVn) + (1.0/h**sigma) * ufl.inner(ufl.dot(tV, n), ufl.dot(TV, n)) ) * ds_cyl
# a12 += (+ ufl.inner(tP, ufl.dot(TV, n))) * ds_cyl
# a21 += (+ ufl.inner(TP, ufl.dot(tV, n))) * ds_cyl

# ------------------------------
# 6) Assemble & solve (nest)
# ------------------------------
A = fem.petsc.assemble_matrix_nest([[a11, a12],
                                    [a21, a22]], bcs=bcs)
A.assemble()
b = fem.petsc.assemble_vector_nest([L1, L2])
fem.petsc.apply_lifting_nest(b, [[a11, a12], [a21, a22]], bcs=bcs)
for bi in b.getNestSubVecs():
    bi.ghostUpdate(addv=PETSC.InsertMode.ADD, mode=PETSC.ScatterMode.REVERSE)
fem.petsc.set_bc_nest(b, bcs)

# KSP solve: monolithic 2×2 nest with LU (switch to iterative for scale)
ksp = PETSc.KSP().create(MPI.COMM_WORLD)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")  # if available
x = A.createVecRight()
ksp.solve(b, x)

# Extract sub-vectors to Functions
u = fem.Function(V); p = fem.Function(Q)
u.x.array[:] = x.getNestSubVecs()[0].array_r
p.x.array[:] = x.getNestSubVecs()[1].array_r

# ------------------------------
# 7) Quick diagnostics: free-slip proof on cylinder
# ------------------------------
ds_c = ds_cyl
un2 = fem.assemble_scalar(fem.form(ufl.dot(u, n)**2 * ds_c))
Ut2 = fem.assemble_scalar(fem.form(ufl.inner(u, u) * ds_c))
if MPI.COMM_WORLD.rank == 0:
    print("|u·n|_L2(cyl) =", np.sqrt(un2), "  rel =", np.sqrt(un2)/(np.sqrt(Ut2)+1e-16))

# ------------------------------
# 8) (Optional) save to XDMF
# ------------------------------
with io.XDMFFile(MPI.COMM_WORLD, "stokes_u.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u)
with io.XDMFFile(MPI.COMM_WORLD, "stokes_p.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(p)
