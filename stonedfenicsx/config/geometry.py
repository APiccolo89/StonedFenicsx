from dataclasses import dataclass, field
import dolfinx
import numpy as np
from numpy.typing import NDArray
from mpi4py import MPI

#------------------------------------------------------------------------------------------------
@dataclass(slots=True)
class Domain:
    """
    Domain object storing the mesh and all associated metadata.

    This dataclass represents either the full computational domain (global mesh)
    or one of its subdomains (e.g., wedge, subducting plate, overriding plate).

    It provides the necessary information to transfer data between the global mesh
    and extracted submeshes, ensuring consistent handling of markers, facets,
    material phases, and boundary conditions.

    Attributes
    ----------
    hierarchy : str
        Mesh hierarchy level:
        - `"parent"` for the global mesh
        - `"child"` for a submesh

    cell_par : NDArray[np.int32] | None
        Parent cell relationships mapping submesh cells to the global mesh cells.
        Only defined if the domain is a submesh.

    node_par : NDArray[np.int32] | None
        Parent node relationships mapping submesh nodes to the global mesh nodes.
        Only defined if the domain is a submesh.

    facets : dolfinx.mesh.MeshTags | None
        Tagged facet markers representing boundary features
        (e.g., trench, free surface, inflow/outflow).

    Tagcells : dolfinx.mesh.MeshTags | None
        Tagged cell markers representing physical regions/material domains.

    bc_dict : dict
        Dictionary mapping boundary condition names to integer tags.

    solPh : dolfinx.fem.FunctionSpace | None
        Function space used to define material property fields or phase functions.

    phase : dolfinx.fem.Function | None
        Material phase indicator function defined on the domain.

    Notes
    -----
    The `Domain` class is a lightweight container for all domain-specific mesh data.
    It allows safe communication of field variables, markers, and boundary tags
    between the global mesh and its corresponding subdomains.
    """

    hierarchy: str = "Parent"
    mesh: dolfinx.mesh.Mesh = None
    cell_par: NDArray[np.int32] = field(default_factory=lambda: np.array([], dtype=np.int32))
    node_par: NDArray[np.int32] = field(default_factory=lambda: np.array([], dtype=np.int32))
    facets: dolfinx.mesh.MeshTags = None
    tagcells: dolfinx.mesh.MeshTags = None
    bc_dict: dict = field(default_factory=dict)
    solph: dolfinx.fem.FunctionSpace = None
    phase: dolfinx.fem.Function = None
#---------------------------------------------------------------------------------------------------
@dataclass(slots=True)
class GeomInput:
    """
    Geometric input parameters defining the subduction setup.
    Lengths in [km]; angles in [degrees]; lc dimensionless.
    ...
    """
    x: NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, 660.0]))
    y: NDArray[np.float64] = field(default_factory=lambda: np.array([-600.0, 0.0]))
    slab_tk: float = 130.0
    cr: float = 30.0
    ocr: float = 7.0
    lit_mt: float = 20.0
    lc: float = 0.3                    # adimensionale, invariato
    ns_depth: float = 50.0
    decoupling: float = 80.0
    resolution_normal: float = 2.0
    resolution_refine: float = 2.0
    theta_out_slab: float = 45.0       # gradi, invariato
    theta_in_slab: float = 10.0        # gradi, invariato
    transition: float = 10.0
    lab_d: float = 100.0
    slab_type: str = "Custom"
    sub_path: str = "Not_Defined"
    sub_lb: float = 300.0
    sub_constant_flag: bool = False
    sub_theta0: float = 5.0           # gradi, invariato
    sub_theta_max: float = 45.0        # gradi, invariato
    sub_trench: float = 0.0
    sub_dl: float = 1.0
    wz_tk: float = 2.0
    van_keken : bool = True

    def check_class_consistency(self):
        """Check the integrity of the input, and if it respect 
        the requirements
        Convert the main bending angle into radians for computation.
        Raises:
            ValueError: Check if the lower crust fraction is a float between 0-1
            ValueError: Check if the sub_type is something that is reasonable
            ValueError: Check if the benchmark mode has the required flags activated
        """
        if not 0.0 <= self.lc <= 1.0:
            raise ValueError(f"lc (lower crust fraction) must be in [0, 1], got {self.lc}")
        if self.slab_type not in {"Custom", "Real"}:
            raise ValueError(f'sub_type must be "Custom" or "Real", got {self.slab_type!r}')
        if self.van_keken and not self.sub_constant_flag:
            raise ValueError('Van Keken benchmark suite requires that the angle flag constant is true')
        if self.slab_type == 'Real' and self.sub_path in ('Not_Defined', None):
            raise ValueError('If you want to test a realistic geometry, why would you not inform me on its location?')
        # Convert the angles in radians
        self.sub_theta0 = np.radians(self.sub_theta0)
        self.sub_theta_max = np.radians(self.sub_theta_max)

#---------------------------------------------------------------------------------------------------
@dataclass(slots=True)
class Mesh:
    """
    Mesh wrapper storing the global mesh, subdomains, and finite element definitions.

    This dataclass acts as a central container for all mesh-related objects used in
    the simulation. It includes the geometric input parameters, the global domain,
    its associated subdomains, and the finite element definitions required for the
    numerical discretization of pressure, temperature, and velocity.

    Attributes
    ----------
    g_input : Geom_input
        Geometric input parameters defining the model setup.

    domainG : Domain
        Global computational domain (full mesh).

    domainA : Domain
        Subduction zone domain (submesh extracted from the global mesh).

    domainB : Domain
        Wedge domain (submesh extracted from the global mesh).

    domainC : Domain
        Overriding plate domain (submesh extracted from the global mesh).

    rank : int
        MPI rank of the current process.

    size : int
        Total number of MPI processes.

    element_p : ufl.FiniteElement
        Finite element definition for the pressure field.

    element_PT : ufl.FiniteElement
        Finite element definition for the temperature field.

    element_V : ufl.FiniteElement
        Finite element definition for the velocity field.
    """

    g_input : GeomInput    # Geometric input
    global_domain : Domain                                # Domain
    subduction_plate_domain : Domain
    wedge_domain : Domain
    crust_domain : Domain
    comm : MPI.Intracomm
    rank : int
    element_p  : object
    element_pt : object
    element_v  : object
#-----------------------------------------------------------------------------------------------------------------