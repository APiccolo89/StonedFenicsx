from typing import get_type_hints, get_origin, get_args, Callable
from dataclasses import dataclass, field
import numpy as np
import mpi4py.MPI as MPI
import time as timing 
import ufl 
import dolfinx
from functools import wraps
from numpy.typing import NDArray
# ---------------------------------------------------------------------------------------------------------
def timing_function(fun: Callable) -> Callable:
    """Extract the execution time of the function.

    Args:
        fun (Callable): The function for which the time of execution needs to be evaluated

    Returns:
        (callable): the output of the given **fun**.
    """

    @wraps(fun)
    def wrapper(*args, **kwargs):
        comm = MPI.COMM_WORLD
        time_a = timing.time()
        result = fun(*args, **kwargs)
        time_b = timing.time()
        dt = time_b - time_a
        dt = comm.allreduce(dt, op=MPI.MAX)
        if comm.rank == 0:
            if dt > 60.0:
                m, s = divmod(dt, 60)
                print(f".  {fun.__name__} took {m:.2f} min and {s:.2f} sec")
            if dt > 3600.0:
                m, s = divmod(dt, 60)
                h, m = divmod(m, 60)
                print(
                    f".  {fun.__name__} took {dt/3600:.2f} hr, {m:.2f} min and {s:.2f} sec"
                )
            else:
                print(f".  {fun.__name__} took {dt:.2f} sec")
        return result

    return wrapper


def time_the_time(delta_time: float) -> float:
    """_summary_

    Args:
        delta_time (float): delta time -> time of execution

    Returns:
        global_dt(float): the effective time accross all the processors.
    """
    comm = MPI.COMM_WORLD
    global_dt = comm.allreduce(delta_time, op=MPI.MAX)
    return global_dt


# ---------------------------------------------------------------------------------------------------------
def print_ph(string: str) -> int:
    """function to print information. Print information only in one processor.
    Args:
        string (str): string to print

    Returns:
        int:
    """
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print(string)
        return 0
    return -1


def interpolate_from_sub_to_main(u_dest: dolfinx.fem.Function
                                 , u_start: dolfinx.fem.Function,
                                 cells: np.ndarray,
                                 parent2child: int = 0) -> dolfinx.fem.Function:
    """
    Interpolate the solution from the subdomain to the main domain.

    Parameters:
        u_slab (Function): The solution in the subdomain.
        u_global (Function): The solution in the main domain.
        M (Mesh): The mesh of the main domain.
        V (FunctionSpace): The function space of the main domain.
    """
    if parent2child == 0:
        a = np.arange(len(cells))
        b = cells
    else:
        a = cells
        b = np.arange(len(cells))

    u_dest.interpolate(u_start, cells0=a, cells1=b)

    return u_dest


def gather_vector(v):
    """_summary_

    Args:
        v (function): vector/function space to syncronise from local to global

    Returns:
        _type_:
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Suppose v is fem.Function(V) (not the UFL test function!)
    v.x.scatter_forward()  # update ghosts from owners

    # Get number of owned dofs (exclude ghosts)
    imap = v.function_space.dofmap.index_map
    n_owned = imap.size_local

    # Slice only owned part
    lv = np.asarray(v.x.array[:n_owned], dtype=np.float64)  # ensure dtype/contiguous

    # Gather element counts on root
    sizes = comm.gather(lv.size, root=0)

    if rank == 0:
        counts = np.asarray(sizes, dtype="i")
        displs = np.insert(np.cumsum(counts[:-1]), 0, 0).astype("i")
        gv = np.empty(int(counts.sum()), dtype=np.float64)
    else:
        counts = None
        displs = None
        gv = None

    # Gather variable-length arrays; only root provides recv buffers/metadata
    comm.Gatherv(lv, (gv, counts, displs, MPI.DOUBLE), root=0)

    if rank == 0:
        return lv
    else:
        return gv  # TO CHECK!!!!!


def gather_coordinates(V):
    """
    Gather DOF coordinates for a dolfinx FunctionSpace V to rank 0.
    Returns an (ndofs_global, gdim) array on rank 0, else None.
    """
    comm = V.mesh.comm
    gdim = V.mesh.geometry.dim

    # Local coordinates (shape: n_local_total x gdim). Slice to owned DOFs only.
    coords_local = V.tabulate_dof_coordinates()
    n_owned = V.dofmap.index_map.size_local
    coords_owned = coords_local[:n_owned, :]

    # Flatten to 1D buffer for Gatherv
    sendbuf = np.ascontiguousarray(coords_owned.ravel(), dtype=np.float64)

    # Gather row counts, then convert to element counts by * gdim
    rows_local = coords_owned.shape[0]
    rows_counts = comm.gather(rows_local, root=0)

    if comm.rank == 0:
        elem_counts = np.asarray(rows_counts, dtype="i") * gdim  # elements, not bytes
        elem_displs = np.insert(np.cumsum(elem_counts[:-1]), 0, 0).astype("i")
        recvbuf = np.empty(int(elem_counts.sum()), dtype=np.float64)
    else:
        elem_counts = None
        elem_displs = None
        recvbuf = None

    # Gather – use MPI.DOUBLE (not a NumPy dtype)
    comm.Gatherv(sendbuf, (recvbuf, elem_counts, elem_displs, MPI.DOUBLE), root=0)

    if comm.rank == 0:
        return recvbuf.reshape(-1, gdim)
    else:
        return None


# ----------------------------------------------------------------------------
def compute_strain_rate(u):
    """Compute strain rate from the velocity field u.

    Args:
        u (function): velocity field
    Returns:
        e (function): strain rate field
    """
    e = ufl.sym(ufl.grad(u))

    return e


# ---------------------------------------------------------------------------


def compute_eii(e):
    """Compute the second invariant of the strain rate from the strain rate field.

    Args:
        e (function): strain rate field

    Returns:
        e_ii (function): second invariant of the deviatoric strain rate
    """
    e_ii = ufl.sqrt(0.5 * ufl.inner(e, e))
    return e_ii




def evaluate_material_property(
    expression:dolfinx.fem.Expression, function_space:dolfinx.fem.FunctionSpace
) -> dolfinx.fem.Function:
    """Transform an ufl expression into a function
    Args:
        expression (ufl.Expression): ufl expression
        function_space (dolfinx.fem.FunctionSpace): the function space
    Returns:
        target_function: The final function
    """

    target_function = dolfinx.fem.Function(function_space)
    target_function.interpolate(
        dolfinx.fem.Expression(expression, function_space.element.interpolation_points())
    )
    return target_function

