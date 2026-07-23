# --- 
from __future__ import annotations
import numpy as np 
import dolfinx.fem as fem 
import dolfinx 
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from numpy.typing import NDArray
from typing import TYPE_CHECKING
from dataclasses import dataclass, field, InitVar
# --- 
from stonedfenicsx.config.scal import Scal
from stonedfenicsx.config.numerical_control import SimulationControls
from stonedfenicsx.config.geometry import GeomInput
from stonedfenicsx.utils import print_ph, timing

if TYPE_CHECKING:
    from stonedfenicsx.solver_module.problems_solution import Solution

# ---
# ---
@dataclass(slots=True)
class OUTERITERATION_SOL_VAL:
    """Class that handles the outer iteration variables
    """
    
    sol: InitVar[Solution]
    T        : dolfinx.fem.function =  field(init=False)
    PL       : dolfinx.fem.function =  field(init=False)
    u : dolfinx.fem.function =  field(init=False)
    p : dolfinx.fem.function =  field(init=False)
    mom_res_wedge : NDArray[float]=  field(init=False)
    mom_res_slab : NDArray[float]=  field(init=False)
    div_res_slab : NDArray[float] =  field(init=False)
    div_res_wedge : NDArray[float] =  field(init=False)
    ene_res_gl : NDArray[float] =  field(init=False)
    res: NDArray[float] = 1.0 
    def __post_init__(self,sol):
        self.T = sol.T_N.copy()
        self.T.x.scatter_forward()
        self.PL = sol.PL.copy()
        self.PL.x.scatter_forward()
        self.u = sol.u_global.copy()
        self.u.x.scatter_forward()
        self.p = sol.p_global.copy()
        self.p.x.scatter_forward()
        #--- specific to domain : momentum 
        self.mom_res_slab =np.zeros(2)
        self.mom_res_wedge = np.zeros(2)
        #---                    : mass conservation 
        self.div_res_slab = np.zeros(2)
        self.div_res_wedge = np.zeros(2)
        
        self.ene_res_gl = np.zeros(2)
    
    def update_iteration(self,sol): 
        self.T.x.array[:] = sol.T_N.x.array[:]
        self.T.x.scatter_forward()
        self.PL.x.array[:] = sol.PL.x.array[:]
        self.PL.x.scatter_forward()
        self.u.x.array[:] = sol.u_global.x.array[:]
        self.u.x.scatter_forward() 
        self.p.x.array[:] = sol.p_global.x.array[:]
        self.p.x.scatter_forward()
           
    def compute_residuum_outer(self
                               ,sol:Solution
                               ,it_outer:int
                               ,sc:Scal
                               ,tA:float
                               ,ts:int
                               ,ctrl_sim:SimulationControls
                               ) -> tuple[float]:
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

        res_u = compute_residuum(sol.u_global,self.u)
        res_p = compute_residuum(sol.p_global,self.p)
        res_T = compute_residuum(sol.T_N,self.T)
        res_PL= compute_residuum(sol.PL,self.PL)

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
            print_ph(' WARNING:::Temperature higher than the maximum temperature')
        if minMaxT[0] < 0.0: 
            print_ph("Problem with the thermal solver")


        res_total = np.sqrt(res_T**2+res_p**2+res_u**2+res_PL**2)
        if not np.isfinite(res_total):
            raise ValueError("res_total is NaN/Inf; check inputs and residual computations.")

        time_B_outer = timing.time()

        print_ph('        --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ')
        print_ph('         Residual difference function :')
        print_ph(f'            []Res velocity       =  {res_u:.3e} [n.d.], max = {minMaxU[1]:.3f}, min = {minMaxU[0]:.3f} [cm/yr], RMS = {minMaxU[2]:.3f} [n.d]')
        print_ph(f'            []Res Temperature    =  {res_T:.3e} [n.d.], max = {minMaxT[1]:.3f}, min = {minMaxT[0]:.3f} [C], RMS = {minMaxT[2]:.3f} [n.d.] ')
        print_ph(f'            []Res pressure       =  {res_p:.3e} [n.d.], max = {minMaxP[1]:.3e}, min = {minMaxP[0]:.3e} [GPa]')
        print_ph(f'            []Res lithostatic    =  {res_PL:.3e}[n.d.], max = {minMaxPL[1]:.3e}, min = {minMaxPL[0]:.3e} [GPa]')
        print_ph(f'            []Res total (sqrt(rT^2+rp^2+ru^2+rPL^2)) =  {res_total:.3e} [n.d.] ')
        print_ph('         Conservation residual :')
        print_ph('          Stokes Equation :')
        a = self.mom_res_wedge[0] * sc.force/sc.length**3
        b = self.mom_res_slab[0] * sc.force/sc.length**3
        c = self.div_res_wedge[0] * sc.strain_rate 
        d = self.div_res_slab[0] * sc.strain_rate
        e = self.ene_res_gl[0] * sc.watt/sc.length**3
        print_ph(f'            []Res mom wedge = abs: {self.mom_res_wedge[0]:.3e} [n.d],{a:.3e} [N/m3] | rel: {self.mom_res_wedge[0]/self.mom_res_wedge[1]:.3e} [n.d.]')
        print_ph(f'            []Res mom_slab  = abs: {self.mom_res_slab[0]:.3e} [n.d],{b:3e} [N/m3] | rel: {self.mom_res_slab[0]/self.mom_res_slab[1]:.3e} [n.d.]')
        print_ph(f'            []Res div_wedge = abs: {self.div_res_wedge[0]:.3e} [n.d],{c:3e} [1/s] | rel: {self.div_res_wedge[0]/self.div_res_wedge[1]:.3e} [n.d.]') 
        print_ph(f'           []Res div_slab  = abs: {self.div_res_slab[0]:.3e} [n.d],{d:3e} [1/s] | rel: {self.div_res_slab[0]/self.div_res_slab[1]:.3e} [n.d.]') 
        print_ph('          Energy Equation :')
        print_ph(f'            []Res energy equation = abs: {self.ene_res_gl[0]:.3e} [n.d],{e**3:3e} [1/s] | rel: {self.ene_res_gl[0]/self.ene_res_gl[1]:.3e} [n.d.]')
        r_tot_conv = np.sqrt(self.mom_res_wedge[0]**2+self.mom_res_slab[0]**2+self.div_res_slab[0]**2+self.div_res_wedge[0]**2+self.ene_res_gl[0]**2)
        r_tot_rel  = np.sqrt(self.mom_res_wedge[1]**2+self.mom_res_slab[1]**2+self.div_res_slab[1]**2+self.div_res_wedge[1]**2+self.ene_res_gl[1]**2)

        print_ph(f'         Combined residual =  abs {r_tot_conv:.3e} [n.d.], rel {r_tot_conv/r_tot_rel:.3e}')
        print_ph('        --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ')

        print_ph(f'   --- Outer iteration {it_outer:d} with tolerance {res_total:.3e}, in {time_B_outer-tA:.1f} sec -- ---')

        # Update solution 
        update_solution(sol.T_N,self.T,ctrl_sim.ctrl.relax)
        sol.T_N.x.array[:] = self.T.x.array[:]
        update_solution(sol.u_global,self.u,ctrl_sim.ctrl.relax)
        sol.u_global.x.array[:] = self.u.x.array[:]
        update_solution(sol.p_global,self.p,ctrl_sim.ctrl.relax)
        sol.p_global.x.array[:] = self.p.x.array[:]


        sol.mT.append(minMaxT[0])
        sol.MT.append(minMaxT[1])
        sol.RMST.append(minMaxT[2])

        sol.mv.append(minMaxU[0])
        sol.Mv.append(minMaxU[1])
        sol.RMSv.append(minMaxU[2])

        sol.outer_iteration.append(res_total)
        sol.ts.append(ts)
        # Update the structure, update the residual
        self.res = res_total 
        self.update_iteration(sol)
        
# ---
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
    
    
    
    # Use explicit duplicate/axpy + destroy: the `+`/`-` operator overloads on
    # PETSc Vec allocate a new temporary Vec each call that is never freed,
    # which leaks across every outer Picard iteration if left to the operators.
    diff = a.x.petsc_vec.copy()
    diff.axpy(-1.0, b.x.petsc_vec)  # diff = a - b
    res = diff.norm(PETSc.NormType.NORM_2)
    diff.destroy()

    total = a.x.petsc_vec.copy()
    total.axpy(1.0, b.x.petsc_vec)  # total = a + b
    dxa = total.norm(PETSc.NormType.NORM_2)
    total.destroy()

    return res / dxa
# ---
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
        # `.sub(i).collapse()` builds a brand-new FunctionSpace (dofmap,
        # index map, ghost layout) every call. Since the vector field is
        # stored blocked/interleaved, the components can be read directly
        # off the owned range without collapsing - this was previously
        # rebuilding two FunctionSpaces every single outer iteration.
        num_owned = a.function_space.dofmap.index_map.size_local
        bs = a.function_space.dofmap.index_map_bs
        owned = a.x.array[:num_owned * bs].reshape(-1, bs)
        array = np.sqrt(owned[:, 0]**2 + owned[:, 1]**2)
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
# ---
def update_solution(sk1:dolfinx.fem.function.Function
                    ,sk0:dolfinx.fem.function.Function
                    ,tol:float)->None:
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
# ---       
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
# ---          
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

