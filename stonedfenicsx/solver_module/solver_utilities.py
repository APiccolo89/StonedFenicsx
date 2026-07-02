# --- 
from __future__ import annotations
import numpy as np 
import dolfinx.fem as fem 
import dolfinx 
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from numpy.typing import NDArray

# --- 
from stonedfenicsx.config.scal import Scal
from stonedfenicsx.config.numerical_control import SimulationControls
from stonedfenicsx.config.geometry import GeomInput
from stonedfenicsx.utils import print_ph, timing
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from stonedfenicsx.solver_module.problems_solution import Solution

#-------------------------------------------------------------------------       
def decoupling_function(z: np.ndarray, fun: dolfinx.fem.Function, g_input: GeomInput)-> dolfinx.fem.Function:
    """Compute the slab–wedge decoupling weight field from depth coordinates.

    Fills `fun` with a smooth tanh ramp that transitions from 1 (fully
    coupled, above the decoupling depth) to 0 (fully decoupled, below it).
    The transition width is one quarter of `g_input.transition`.  The result
    is normalised by `g_input.decoupling` so the function is dimensionless
    and correct whether coordinates are physical or non-dimensionalised.

    Args:
        z (np.ndarray): Depth coordinate array at every DOF, in the same
            unit as the mesh (non-dimensionalised in normal use).
        fun (dolfinx.fem.Function): Pre-allocated scalar Function on the
            sub-mesh where the weight is needed; modified in-place.
        g_input (GeomInput): Geometric input holding decoupling depth
            (`decoupling`), total lithosphere depth (`ns_depth`), and the
            transition half-width (`transition`).

    Returns:
        dolfinx.fem.Function: The same `fun` object, updated with the
        decoupling weight and scatter-forwarded for MPI consistency.
    """
    
    dc = g_input.decoupling
    lit = g_input.ns_depth
    dc = dc/g_input.decoupling
    lit = lit/g_input.decoupling
    z2 = np.abs(z)/g_input.decoupling
    trans = g_input.transition/g_input.decoupling
    

    fun.x.array[:] = 1-0.5 * ((1.0)+(1.0)*np.tanh((z2-dc)/(trans/4)))
    # Parallel operation 
    fun.x.scatter_forward()
    
    
    
    return fun

#---------------------------------------------------------------------------------------------------          

def L2_norm_calculation(f:dolfinx.fem.Function) -> float:
    """Compute the global L2 norm of a fem.Function over its mesh.

    Assembles the local contribution of (f, f) on each MPI rank and reduces
    across all ranks before taking the square root, so the result is correct
    in parallel.

    Args:
        f (dolfinx.fem.Function): Scalar or vector fem.Function defined on
            a dolfinx mesh.

    Returns:
        float: ||f||_L2 = sqrt( integral f·f dx ) over the full (parallel) mesh.
    """
    comm = f.function_space.mesh.comm
    local = fem.assemble_scalar(fem.form(ufl.inner(f, f) * ufl.dx))
    global_sq = comm.allreduce(local, op=MPI.SUM)
    return np.sqrt(global_sq)

def compute_residuum_outer(sol:Solution
                           ,T:dolfinx.fem.Function
                           ,PL:dolfinx.fem.Function
                           ,u:dolfinx.fem.Function
                           ,p:dolfinx.fem.Function
                           ,it_outer:int
                           ,sc:Scal
                           ,tA:float
                           ,ts:int
                           ,ctrl_sim:SimulationControls
                           ) -> tuple[float,Solution]:
    """Compute the outer-loop Picard residual and print diagnostic statistics.

    Computes a normalised L2 residual for each of the four solution fields
    (velocity, dynamic pressure, temperature, lithostatic pressure) between
    the current and previous outer iterations.  The combined residual is the
    Euclidean norm of the four individual residuals.  Physical-unit ranges
    (min, max, RMS) are rescaled for printing.

    Appends per-timestep diagnostics (min/max T, min/max v, RMS, outer
    residual, timestep index) to the history arrays stored in `sol`.

    Raises ValueError if the combined residual is non-finite, which signals
    a diverged or ill-conditioned solve.

    Args:
        sol (Solution): Current solution container (fields read, history
            arrays appended in-place).
        T (dolfinx.fem.Function): Temperature at the start of this outer
            iteration (snapshot copy made by `outerloop_operation`).
        PL (dolfinx.fem.Function): Lithostatic pressure at the start of this
            outer iteration.
        u (dolfinx.fem.Function): Velocity at the start of this outer
            iteration.
        p (dolfinx.fem.Function): Dynamic pressure at the start of this outer
            iteration.
        it_outer (int): Current outer-loop iteration index (for printing).
        sc (Scal): Non-dimensionalisation scaling object for unit rescaling.
        tA (float): Wall-clock time (from `timing.time()`) at the start of
            the outer iteration, used to report elapsed time.
        ts (int): Current timestep index.
        ctrl_sim (SimulationControls): Simulation controls; used to check
            whether temperature has exceeded the prescribed maximum.

    Returns:
        tuple[float, Solution]:
            res_total -- combined outer-loop residual (dimensionless).
            sol       -- solution container with updated history arrays.
    """
    # Prepare the variables 
    
    res_u = compute_residuum(sol.u_global,u)
    res_p = compute_residuum(sol.p_global,p)
    res_T = compute_residuum(sol.T_N,T)
    res_PL= compute_residuum(sol.PL,PL)
    
    # Compute the ranges
    minMaxU = min_max_array(sol.u_global, vel=True)
    minMaxP = min_max_array(sol.p_global)
    minMaxT = min_max_array(sol.T_N)
    minMaxPL= min_max_array(sol.PL)
    
    # scal back 
    
    minMaxU[0:2] = minMaxU[0:2]*(sc.length/sc.time)/sc.scale_vel 
    minMaxP = minMaxP*sc.stress/1e9 
    minMaxT[0:2] = minMaxT[0:2]*sc.temp -273.15
    minMaxPL = minMaxPL*sc.stress/1e9
 
    
    if minMaxT[1]-(ctrl_sim.ctrl_tbc.temp_max * sc.temp-273.15)>1.0: 
        print_ph('Temperature higher than the maximum temperature')
    if minMaxT[0] < 0.0: 
        print_ph("Problem with the thermal solver")

    
    res_total = np.sqrt(res_T**2+res_p**2+res_u**2+res_PL)
    if not np.isfinite(res_total):
        raise ValueError("res_total is NaN/Inf; check inputs and residual computations.")
    
    time_B_outer = timing.time()

    print_ph('')
    print_ph(f' Outer iteration {it_outer:d} with tolerance {res_total:.3e}, in {time_B_outer-tA:.1f} sec // -- // --->')
    print_ph(f'    []Res velocity       =  {res_u:.3e} [n.d.], max = {minMaxU[1]:.6f}, min = {minMaxU[0]:.6f} [cm/yr], RMS = {minMaxU[2]:.6f} [n.d]')
    print_ph(f'    []Res Temperature    =  {res_T:.3e} [n.d.], max = {minMaxT[1]:.6f}, min = {minMaxT[0]:.6f} [C], RMS = {minMaxT[2]:.6f} [n.d.] ')
    print_ph(f'    []Res pressure       =  {res_p:.3e} [n.d.], max = {minMaxP[1]:.3e}, min = {minMaxP[0]:.3e} [GPa]')
    print_ph(f'    []Res lithostatic    =  {res_PL:.3e}[n.d.], max = {minMaxPL[1]:.3e}, min = {minMaxPL[0]:.3e} [GPa]')
    print_ph(f'    []Res total (sqrt(rT^2+rp^2+ru^2+rPL^2)) =  {res_total:.3e} [n.d.] ')
    print_ph('. =============================================// -- // --->')
    print_ph('')
    
    sol.mT = np.append(sol.mT,minMaxT[0])
    sol.MT = np.append(sol.MT,minMaxT[1])
    sol.RMST = np.append(sol.RMST,minMaxT[2])
    
    sol.mv = np.append(sol.mv,minMaxU[0])
    sol.Mv = np.append(sol.Mv,minMaxU[1])
    sol.RMSv = np.append(sol.RMSv,minMaxU[2])

    sol.outer_iteration = np.append(sol.outer_iteration,res_total)
    sol.ts = np.append(sol.ts,ts)
    
    
    return res_total, sol 

#------------------------------------------------------------------------------------------------------------

def compute_residuum(a:dolfinx.fem.Function,b:dolfinx.fem.Function)->float:
    """Compute the normalised PETSc-vector residual between two solution fields.

    Uses PETSc NORM_2 on the underlying PETSc vectors to evaluate:
        res = ||a - b||_2 / ||a + b||_2
    which is a relative change norm immune to absolute scaling of the fields.
    The norms are computed inside PETSc and are already MPI-global.

    Args:
        a (dolfinx.fem.Function): Current (new) solution field.
        b (dolfinx.fem.Function): Previous (old) solution field; must live on
            the same function space as `a`.

    Returns:
        float: Dimensionless relative residual in [0, inf).
    """
    
    
    
    dxa = (a.x.petsc_vec+b.x.petsc_vec).norm(PETSc.NormType.NORM_2)
    res = (a.x.petsc_vec - b.x.petsc_vec).norm(PETSc.NormType.NORM_2)  / dxa
    
    return res
#------------------------------------------------------------------------------------------------------------
def min_max_array(a:dolfinx.fem.function.Function
                ,vel = False)->NDArray[np.float64]:
    """Compute global min, max, and volume-averaged RMS of a field.

    For scalar fields the min/max are taken over owned DOFs only (ghost DOFs
    excluded) before the MPI allreduce, preventing double-counting on shared
    nodes.  For vector fields (vel=True) the magnitude is formed from the x
    and y sub-components before computing min/max.  The RMS is the
    volume-normalised L2 norm: ||a||_L2 / sqrt(volume).

    Args:
        a (dolfinx.fem.function.Function): Scalar or 2-D vector fem.Function.
        vel (bool, optional): If True, treat `a` as a 2-D velocity vector and
            compute the pointwise speed magnitude. Defaults to False.

    Returns:
        NDArray[np.float64]: Shape-(3,) array [global_min, global_max, RMS],
        all in the non-dimensionalised units of `a`.
    """
    
    if vel: 
        a1 = a.sub(0).collapse()
        num_owned = a1.function_space.dofmap.index_map.size_local

        a2 = a.sub(1).collapse()
        num_owned2 = a2.function_space.dofmap.index_map.size_local

        array = np.sqrt(a1.x.array[:num_owned]**2 + a2.x.array[:num_owned2]**2)
    else: 
        num_owned = a.function_space.dofmap.index_map.size_local
        a.x.scatter_forward()
        array = a.x.array[:]
        array = array[:num_owned]
        
    
    local_min = np.min(array[:])
    local_max = np.max(array[:])
    
    global_min = a.function_space.mesh.comm.allreduce(local_min, op=MPI.MIN)
    global_max = a.function_space.mesh.comm.allreduce(local_max, op=MPI.MAX)
    
    # Compute the L2 norm  https://jsdokken.com/FEniCS23-tutorial/src/benefits_of_curved_meshes.html
    dx = ufl.dx(domain=a.function_space.mesh)
    L2_na = L2_norm_calculation(a)
    volume = fem.assemble_scalar(fem.form(1.0 * dx))
    volume_int = a.function_space.mesh.comm.allreduce(volume,MPI.SUM)
    RMS = L2_na/np.sqrt(volume_int) 
    
    return np.array([global_min, global_max, RMS],dtype=np.float64)


def update_solution(sk1:dolfinx.fem.function.Function
                    ,sk0:dolfinx.fem.function.Function
                    ,tol:float)->dolfinx.fem.function.Function:
    """Apply under-relaxation to a solution field.

    Blends the new iterate sk1 with the old iterate sk0 using relaxation
    factor `tol`:
        sk0 ← tol * sk1 + (1 - tol) * sk0

    A value of tol=1 accepts the new iterate fully; tol<1 damps oscillations
    in slowly-converging or marginally-stable Picard problems.  The result is
    written back into sk0 in-place so no new allocation occurs.

    Args:
        sk1 (dolfinx.fem.function.Function): Solution from the current
            inner-loop iteration.
        sk0 (dolfinx.fem.function.Function): Solution from the previous
            inner-loop iteration; overwritten with the relaxed value.
        tol (float): Relaxation factor in (0, 1].

    Returns:
        dolfinx.fem.function.Function: The updated `sk0` (same object).
    """

    
    # extract reference to sk1.x.petsc_vec => modify, then reintegrate
    sk0.x.array[:] = tol * sk1.x.array[:] + (1-tol) * sk0.x.array[:]

    sk0.x.scatter_forward()
    
    return sk0

