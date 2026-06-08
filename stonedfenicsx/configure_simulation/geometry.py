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

    cell_par : np.ndarray | None
        Parent cell relationships mapping submesh cells to the global mesh cells.
        Only defined if the domain is a submesh.

    node_par : np.ndarray | None
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
    cell_par: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    node_par: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    facets: dolfinx.mesh.MeshTags = None
    Tagcells: dolfinx.mesh.MeshTags = None
    bc_dict: dict = field(default_factory=dict)
    solPh: dolfinx.fem.FunctionSpace = None
    phase: dolfinx.fem.Function = None
#---------------------------------------------------------------------------------------------------
@dataclass(slots=True)
class Geom_input:
    """
    Geometric input parameters defining the subduction setup.

    This dataclass stores the main geometric quantities required to build the
    computational domain and prescribe the slab geometry.

    Attributes
    ----------
    x : float
        Main grid coordinate in the x-direction (SI units: [m]).
    y : float
        Main grid coordinate in the y-direction (SI units: [m]).
        Can be negative.
    slab_tk : float
        Thickness of the subducting slab (SI units: [m]).
    cr : float
        Thickness of the overriding crust (SI units: [m]).
    ocr : float
        Thickness of the oceanic crust (SI units: [m]).
    lit_mt : float
        Depth of the lithospheric mantle (SI units: [m], always positive).
    lc : float
        Lower crust ratio of the overriding crust (dimensionless, value in [0, 1]).
    ns_depth : float
        Depth of the no-slip boundary condition (SI units: [m], always positive).
    decoupling : float
        Depth of the slab–mantle decoupling (SI units: [m], always positive).
    resolution_normal : float
        Minimum grid resolution (SI units: [m], always positive).
    resolution_refine : float
        Maximum grid refinement resolution (SI units: [m], always positive).
    theta_out_slab : float
        Slab bending angle at the bottom of the simulation domain (degrees).
    theta_in_slab : float
        Slab bending angle at the trench (degrees).
    trans : float
        Transition interval over which coupling/uncoupling occurs (SI units: [m]).
    lab_d : float
        Depth of the lithosphere–asthenosphere boundary (SI units: [m]).
    sub_type : str
        Geometry type, either `"Custom"` (internal geometry) or `"Real"`
        (external geometry database).
    sub_path : str
        Path or URL of the external geometry database (used if `sub_type="Real"`).
    sub_Lb : float
        Along-slab distance where bending occurs (SI units: [m]).
    sub_constant_flag : int
        Flag controlling whether the slab bending angle is constant.
    sub_theta_0 : float
        Initial bending angle at the upper-left corner of the slab (degrees).
    sub_theta_max : float
        Maximum bending angle after the critical distance `sub_Lb` (degrees).
    sub_trench : float
        Horizontal position of the trench (SI units: [m]).
    sub_dl : float
        Segment length used to discretize the slab surface (SI units: [m]).
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]

    slab_tk: float
    cr: float
    ocr: float
    lit_mt: float
    lc: float

    ns_depth: float
    decoupling: float

    resolution_normal: float
    resolution_refine: float

    theta_out_slab: float
    theta_in_slab: float

    trans: float
    lab_d: float

    sub_type: str
    sub_path: str
    sub_Lb: float

    sub_constant_flag: bool
    sub_theta_0 : float 
    sub_theta_max : float
    
    sub_trench : float 
    sub_dl : float
    wz_tk : float 
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

    g_input : Geom_input    # Geometric input
    domainG : Domain                                # Domain
    domainA : Domain                     
    domainB : Domain
    domainC : Domain
    comm : MPI.Intracomm
    rank : int
    element_p  : object   
    element_PT : object
    element_V  : object
     
#-----------------------------------------------------------------------------------------------------------------