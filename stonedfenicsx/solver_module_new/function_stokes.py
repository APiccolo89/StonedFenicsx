import dolfinx 
import ufl
import numpy as np 
import typing 
from stonedfenicsx.solver_module_new.problems_solution_obj import ProblemOBJ, Solutions 
from collections import namedtuple

class fem_form_stokes(namedtuple):
    a : ufl.form.Form
    a_p0 : ufl.form.Form
    linear : ufl.form.Form


def fem_stokes_form(a1,a2,a3,a_p):
    a   = [[a1, a2],[a3, None]]
    a_p0  = [[a1, a2],[a3, a_p]]
    return a,a_p0


def compute_nitsche_free_slip(obj:ProblemOBJ
                       ,sol:Solutions
                       ,a1:ufl.form.Form
                       ,a2:ufl.form.Form
                       ,a3:ufl.form.Form
                       ,gamma:float
                       ,it:int = 0)->tuple([ufl.form.Form,ufl.form.Form,ufl.form.Form]):

    # Compute the shear stress tensor for a given viscosity and velocity field. 
    def tau(eta, u):
        return 2 * eta * ufl.sym(ufl.grad(u))
    
    u, p  = ufl.TrialFunction(obj.func_space.sub(0).collapse()[0]), ufl.TrialFunction(obj.func_space.sub(1).collapse()[0])
    v, q  = ufl.TestFunction(obj.func_space.sub(0).collapse()[0]),  ufl.TestFunction(obj.func_space.sub(1).collapse()[0])
    
    ds_bd = obj.domain.bc_dict["bot_subduction"]
    
    # Linear 
    e   = compute_strain_rate(sol.u)   
    # Viscosity computation
    eta = compute_viscosity_FX(e,sol.temp_new,sol.,FGS,sc)
    # Extract the facet normal and the cell diameter for the mesh to compute the Nitsche terms.
    n = ufl.FacetNormal(obj.domain.mesh)
    h = ufl.CellDiameter(obj.domain.mesh)
    # Update the forms with the Nitsche terms.
    a1 += (
        - ufl.inner(tau(eta, u), ufl.outer(ufl.dot(v, n) * n, n)) * obj.ds(ds_bd)
        - ufl.inner(ufl.outer(ufl.dot(u, n) * n, n), tau(eta, v)) * obj.ds(ds_bd)
        + (2 * eta * gamma / h)
        * ufl.inner(ufl.outer(ufl.dot(u, n) * n, n), ufl.outer(v, n))
        * self.ds(dS)
    )
    if it == 0:
        a2 += ufl.inner(p, ufl.dot(q, n)) * self.ds(dS)
        a3 += ufl.inner(q, ufl.dot(p, n)) * self.ds(dS)
    else:
        a2 += 0 
        a3 += 0 
    return a1, a2, a3 
    

def stokes_form(obj:ProblemOBJ,sol:Solutions,it_outer:int,ts:int) -> tuple[dolfinx.fem.Form, dolfinx.fem.Form, dolfinx.fem.Form, dolfinx.fem.Form, dolfinx.fem.Form]:
    
    from stonedfenicsx.material_property.compute_material_property import cell_average_DG0
        
    u, p  = ufl.TrialFunction(obj.func_space.sub(0).collapse()[0]), ufl.TrialFunction(obj.func_space.sub(1).collapse()[0])
    v, q  = ufl.TestFunction(obj.func_space.sub(0).collapse()[0]),  ufl.TestFunction(obj.func_space.sub(1).collapse()[0])
    dx    = ufl.dx
    e = compute_strain_rate(u_s)
    # If we are in the first iteration of the first timestep -> use the default viscosity for creating an initial guess. fem.Constant(M.domainG.mesh, PETSc.ScalarType([0.0, -ctrl.g]))   
    if it == 0 and ts == 0 and slab == 0:
        eta = fem.Constant(D.mesh,PETSc.ScalarType(FR.eta_def))
    else: 
        eta = compute_viscosity_FX(e,sol.temp_new,sol.p_dyn,obj.cached_mat)
        eta_av = cell_average_DG0(D.mesh, eta)
    
    a1 = ufl.inner(2*eta*ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * dx
    a2 = - ufl.inner(ufl.div(v), p) * dx             # build once
    a3 = - ufl.inner(q, ufl.div(u)) * dx             # build once
    a_p0 =  -1/eta * ufl.inner( q, p) * dx                      # pressure mass (precond)
    f  = fem.Constant(obj.domain.mesh, PETSc.ScalarType((0.0,)*obj.domain.mesh.geometry.dim))
    f2 = fem.Constant(obj.domain.mesh, PETSc.ScalarType(0.0))
    linear  = fem.form([ufl.inner(f, v)*dx, ufl.inner(f2, q)*dx])    

    return a1, a2, a3 , linear , a_p0   

def construct_form_slab(obj:ProblemOBJ,sol:Solutions,it_outer:int,ts:int)->slab_fem_form:
    
    a1, a2, a3, linear, a_p = stokes_form(obj,sol,it_outer,ts)
    a1, a2, a3 = compute_nitsche_free_slip(obj,sol,a1,a2,a3,50.0,it)
    a, a_p0 = fem_stokes_form(a1,a2,a3,a_p)
    slab_fem_form = slab_fem_form(a=a,a_p0=a_p0,linear=linear)
    
    return slab_fem_form
    

    
    
    
    
    

    