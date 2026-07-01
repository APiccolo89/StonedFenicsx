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
    """_summary_

    Args:
        z (np.ndarray): _description_
        fun (dolfinx.fem.Function): _description_
        g_input (GeomInput): _description_

    Returns:
        dolfinx.fem.Function: _description_
    """
    
    dc = g_input.decoupling
    lit = g_input.ns_depth
    dc = dc/g_input.decoupling
    lit = lit/g_input.decoupling
    z2 = np.abs(z)/g_input.decoupling
    trans = g_input.trans/g_input.decoupling
    

    fun.x.array[:] = 1-0.5 * ((1.0)+(1.0)*np.tanh((z2-dc)/(trans/4)))
    # Parallel operation 
    fun.x.scatter_forward()
    
    
    
    return fun

#---------------------------------------------------------------------------------------------------          

def L2_norm_calculation(f:dolfinx.fem.Function) -> float:
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
    
    #sol.T_N = update_solution(sol.T_N,T,ctrl.relax)
    #sol.u_global = update_solution(sol.u_global,u,ctrl.relax)

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
    """Compute the residual of a given solution

    Args:
        a (dolfinx.fem.Function): new solution
        b (dolfinx.fem.Function): old solution

    Returns:
        float: L2 norm residual 
    
    The residual is computed as ||(a-b)/(a+b)|| L2 norm
    """
    
    
    
    dxa = (a.x.petsc_vec+b.x.petsc_vec).norm(PETSc.NormType.NORM_2)
    res = (a.x.petsc_vec - b.x.petsc_vec).norm(PETSc.NormType.NORM_2)  / dxa
    
    return res
#------------------------------------------------------------------------------------------------------------
def min_max_array(a:dolfinx.fem.function.Function
                ,vel = False)->NDArray[np.float64]:
    """Create diagnostic for scalar/vectorial field
    Compute the min and max of the given function, and compute the Root mean square of this field
     
    Args:
        a (dolfinx.fem.function.Function_): Function (i.e., velocity, pressure ... )
        vel (bool, optional): vectorial field. Defaults to False.

    Returns:
        NDArray[np.float64]: array containing min of a, max of a, RMS of a. 
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
                    ,tol:dolfinx.fem.function.Function)->dolfinx.fem.function.Function:
    """Relax the solution 
        us = tol * sk1 + (1-tol) * sk

    Args:
        sk1 (dolfinx.fem.function.Function): new iteration solution
        sk0 (dolfinx.fem.function.Function): old iteration solution
        tol (dolfinx.fem.function.Function): tollerance

    Returns:
        dolfinx.fem.function.Function: _description_
    """

    
    # extract reference to sk1.x.petsc_vec => modify, then reintegrate
    sk0.x.array[:] = tol * sk1.x.array[:] + (1-tol) * sk0.x.array[:]

    sk0.x.scatter_forward()
    
    return sk0

