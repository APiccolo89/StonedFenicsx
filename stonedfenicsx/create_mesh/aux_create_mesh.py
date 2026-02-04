from stonedfenicsx.package_import import *

#---------------------------------------------------------

dict_surf = {
    'sub_plate'         : 1,
    'oceanic_crust'     : 2,
    'wedge'             : 3,
    'overriding_lm'     : 4,
    'upper_crust'       : 5,
    'lower_crust'       : 6,
}
#---------------------------------------------------------


dict_tag_lines = {
    'Top'               : 1,
    'Right_lit'         : 2,
    'Right_wed'         : 3,
    'Bottom_wed'        : 4,
    'Bottom_sla'        : 5,
    'Subduction_bot'    : 6,
    'Left_inlet'        : 7,
    'Subduction_top_lit': 8,
    'Subduction_top_wed': 9,
    'Oceanic'           : 10,
    'Overriding_mantle' : 11,
    'Crust_overplate'   : 12,
    'LCrust_overplate'  : 13,
}

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
class Class_Points():
    def update_points(self,
                      mesh_model : gmsh.model,
                      sx : np.ndarray,
                      sy : np.ndarray,
                      bx : np.ndarray,
                      by : np.ndarray,
                      oc_cx : np.ndarray | None,
                      oc_cy : np.ndarray | None,
                      g_input : Geom_input) -> gmsh.model:

        """
        Generate the physical points required to build the Gmsh geometry.
        
        Using the main geometric information stored in `g_input` together with the slab
        top/bottom surfaces (and optionally the oceanic Moho), this routine defines the
        physical points that will later be connected into physical lines for the Gmsh
        model.
        
        Parameters
        ----------
        sx : np.ndarray
            x-coordinates of the top surface of the subducting plate.
        sy : np.ndarray
            y-coordinates of the top surface of the subducting plate.
        bx : np.ndarray
            x-coordinates of the bottom surface of the subducting plate.
        by : np.ndarray
            y-coordinates of the bottom surface of the subducting plate.
        oc_cx : np.ndarray | None
            x-coordinates of the oceanic Moho. Can be `None`/empty if the crust is not defined.
        oc_cy : np.ndarray | None
            y-coordinates of the oceanic Moho. Can be `None`/empty if the crust is not defined.
        g_input : Geom_input
            Object containing all input geometric information.
        
        Returns
        -------
        mesh_model : gmsh.model
            Updated Gmsh model containing the generated points. 
        -> update the class points with new members 
        """

        # Function CREATE POINTS

        #-- Create the subduction,channel and oceanic crust points 

        self.max_tag_s,  self.tag_subduction, self.coord_sub,     mesh_model     = _create_points(mesh_model, sx,    sy,    g_input.resolution_refine, 0)
        self.max_tag_bots,  self.tag_bottom,    self.coord_bottom, mesh_model     = _create_points(mesh_model, bx,    by,    g_input.resolution_normal, self.max_tag_s)
        if g_input.ocr != 0.0: 
            self.max_tag_oc, self.tag_oc,         self.coord_ocean,   mesh_model     = _create_points(mesh_model, oc_cx, oc_cy, g_input.resolution_normal, self.max_tag_bots)
        else: 
            self.max_tag_oc = self.max_tag_bots; self.coord_ocean  = None
        # -- Here are the points at the boundary of the model. The size of the model is defined earlier, and subduction zone is modified as such to comply the main geometrical input, 
        # I used subduction points because they define a few important point. 


        self.max_tag_b, self.tag_right_c_b, self.coord_bc,  mesh_model            = _create_points(mesh_model,  g_input.x[1], g_input.y[0],         g_input.resolution_normal,  self.max_tag_oc,  True)
        self.max_tag_c, self.tag_right_c_l, self.coord_lr,  mesh_model            = _create_points(mesh_model,   g_input.x[1], -g_input.ns_depth, g_input.resolution_normal,  self.max_tag_b,  True)
        self.max_tag_d, self.tag_right_c_t, self.coord_top, mesh_model            = _create_points(mesh_model,   g_input.x[1],  g_input.y[1],         g_input.resolution_normal,  self.max_tag_c,  True)

        if g_input.cr != 0.0:
            self.max_tag_e, self.tag_right_c_cr, self.coord_crust,    mesh_model  = _create_points(mesh_model,  g_input.x[1], -g_input.cr,                g_input.resolution_normal, self.max_tag_d, True)
            if g_input.lc !=0: 
                    self.max_tag_f, self.tag_right_c_lcr,  self.coord_lcr, mesh_model = _create_points(mesh_model,  g_input.x[1], -g_input.cr*(1-g_input.lc), g_input.resolution_normal, self.max_tag_e, True)
            else: 
                self.coord_lcr = None
        else: 
            self.coord_lcr = None 
            self.coord_crust = None

        # Thank Chatgpt 
        arr = [self.coord_sub, self.coord_bottom, self.coord_ocean, self.coord_bc, self.coord_lr, self.coord_top, self.coord_crust, self.coord_lcr]
        arrays = [a for a in arr if a is not None]

        self.global_points = np.hstack(arrays)

        return mesh_model
#-----------------------------------------------------------------------------------------------------------------
 
class Class_Line():
    def update_lines(self, 
                     mesh_model:gmsh.model, 
                     CP:Class_Points, 
                     g_input:Geom_input)-> gmsh.model:
        """
        Generate Gmsh lines and assign tags/IDs.
        
        This routine creates the physical lines required for the geometry, using the
        previously generated point set stored in `CP`, and assigns consistent tags/IDs
        to each line so they can be referenced later (e.g., for boundary markers and
        physical groups).
        
        Parameters
        ----------
        mesh_model : gmsh.model
            Gmsh model object to be updated.
        CP : Class_Points
            Container storing the physical point tags and their coordinates.
        g_input : Geom_input
            Object containing the geometric input parameters controlling which lines
            are created and how they are connected.
        
        Returns
        -------
        mesh_model : gmsh.model
            Updated Gmsh model containing the generated lines.
        LP : Class_Line
            Updated line container holding the created line IDs/tags and connectivity
            (e.g., mapping line -> endpoint point tags).
        
        Notes
        -----
        The returned line container is used to:
        - build surfaces from line loops,
        - define physical groups for boundaries/regions,
        - and maintain consistent bookkeeping between geometry objects.
        """
        
        # Top Boundary      
        p_list                                                    = [CP.tag_subduction[0], CP.tag_right_c_t[0]]
        self.max_tag_top, self.tag_L_T, self.lines_T, mesh_model  = _create_lines(mesh_model,0,p_list,False)

        #[right boundary]


        if g_input.cr !=0: 
            if g_input.lc !=0:
                p_list = [CP.tag_right_c_t[0],  CP.tag_right_c_lcr[0],  CP.tag_right_c_cr[0],  CP.tag_right_c_l[0], CP.tag_right_c_b[0]]
            else:
                p_list = [CP.tag_right_c_t[0],  CP.tag_right_c_cr[0],   CP.tag_right_c_l[0],   CP.tag_right_c_b[0]]
        else: 
                p_list = [CP.tag_right_c_t[0],  CP.tag_right_c_l[0],    CP.tag_right_c_b[0]]

        self.max_tag_right, self.tag_L_R, self.lines_R, mesh_model  = _create_lines(mesh_model,  self.max_tag_top,  p_list,  False)
        
        #[bottom boundary]
        if g_input.ocr != 0.0:
            p_list                                                      = [CP.tag_right_c_b[0], CP.tag_subduction[-1], CP.tag_oc[-1], CP.tag_bottom[-1]]
        else: 
            p_list                                                      = [CP.tag_right_c_b[0], CP.tag_subduction[-1], CP.tag_bottom[-1]]
        self.max_tag_bottom, self.tag_L_B, self.lines_B, mesh_model = _create_lines(mesh_model, self.max_tag_right, p_list, False)
        
        # ] curved line
        if g_input.ocr != 0.0:
            p_list                                                    = [CP.tag_bottom[0], CP.tag_oc[0], CP.tag_subduction[0]]
        else: 
            p_list                                                    = [CP.tag_bottom[0], CP.tag_subduction[0]]
        self.max_tag_left, self.tag_L_L, self.lines_L, mesh_model = _create_lines(mesh_model,  self.max_tag_bottom,  p_list,  False)
        
        
        # -- Create Lines 
        self.max_tag_line_subduction,  self.tag_L_sub, self.lines_S,   mesh_model      = _create_lines(mesh_model,  self.max_tag_left,           CP.tag_subduction,False)
        self.max_tag_line_Bsubduction, self.tag_L_Bsub,self.lines_BS,  mesh_model      = _create_lines(mesh_model,  self.max_tag_line_subduction,     CP.tag_bottom,False)
        if g_input.ocr != 0.0:
            self.max_tag_line_ocean,       self.tag_L_oc,  self.lines_oc,  mesh_model      = _create_lines(mesh_model,  self.max_tag_line_Bsubduction,   CP.tag_oc,        False)
        else:
            self.lines_oc = None
        # Create line overriding plate:
        #-- find tag of the of the channel # -> find the mistake: confusion with index types
        i_s = find_tag_line(CP.coord_sub,     g_input.ns_depth,'y')
        p_list = [i_s, CP.tag_right_c_l[0]]
        if g_input.ocr != 0.0:
            self.max_tag_line_ov,    self.tag_L_ov,    self.lines_L_ov,  mesh_model         = _create_lines(mesh_model, self.max_tag_line_ocean, p_list, False)
        else:
            self.max_tag_line_ov,    self.tag_L_ov,    self.lines_L_ov,  mesh_model         = _create_lines(mesh_model, self.max_tag_line_Bsubduction, p_list, False)
        if g_input.cr !=0: 

            i_c = find_tag_line(CP.coord_sub, g_input.cr,'y')

            p_list = [i_c, CP.tag_right_c_cr[0]]
            self.max_tag_line_crust, self.tag_L_cr, self.lines_cr, mesh_model             = _create_lines(mesh_model,self.max_tag_line_ov,p_list,False)

            if g_input.lc !=0:

                i_c = find_tag_line(CP.coord_sub,(1-g_input.lc)*g_input.cr,'y')

                p_list = [i_c, CP.tag_right_c_lcr[0]]
                self.max_tag_line_Lcrust, self.tag_L_Lcr, self.lines_lcr, mesh_model      = _create_lines(mesh_model, self.max_tag_line_crust,p_list,False)
            else: 
                self.lines_lcr = None
        else: 
            self.lines_cr = None; self.lines_lcr = None

        arr = [self.lines_T, self.lines_R, self.lines_B, self.lines_L, self.lines_S, self.lines_BS, self.lines_oc, self.lines_L_ov, self.lines_cr, self.lines_lcr]
        arrays = [a for a in arr if a is not None]

        self.line_global = np.hstack(arrays)
        
        return mesh_model
                
#-----------------------------------------------------------------------------------------------------------------
def create_loop(l_list:list,
                mesh_model:gmsh.model,
                tag:int) -> gmsh.model:
    """_summary_

    Args:
        l_list (list): _description_
        mesh_model (gmsh.model): _description_
        tag (int): _description_

    Returns:
        gmsh.model: _description_
    """
    
    
    a = []
    
    for i in range(len(l_list)):
        
        if isinstance(l_list[i], (int, np.integer)): 
            val = [l_list[i]]
        else: 
            val = l_list[i]
    
        a.extend(val)
        
    
    mesh_model.geo.addCurveLoop(a,tag) # Left side of the subudction zone
   
    return mesh_model
#-----------------------------------------------------------------------------------------------------------------
def find_line_index(Lin_ar:NDArray,
                    point:NDArray,
                    d:float) -> int:
    
    for i in range(len(Lin_ar[0,:])-1):
        # Select pint of the given line
        p0 = np.int32(Lin_ar[0,i])

        p1 = np.int32(Lin_ar[1,i])
        # find the index of the point 
        ip0 = np.where(p0==np.int32(point[2,:]))
        ip = np.where(p1==np.int32(point[2,:]))
        # Check wheter or not the coordinate z is the one. 
        z1 = point[1,ip]
        if z1 == -d: 
            X = [point[0,ip0],point[0,ip]]
            Y = [point[1,ip0],point[1,ip]]

            index = i+1 
            break    
    
    
    return index 

#-----------------------------------------------------------------------------------------------------------------

def find_tag_line(coord:NDArray,
                  x:float,
                  dir:str) -> int:
    if dir == 'x':
        a = 0 
    else:
        a = 1 
    
    i = np.where(coord[a,:]==-x)
    i = i[0][0];i = coord[2,i]
    
    return np.int32(i)                 

#-----------------------------------------------------------------------------------------
def find_slab_surface(g_input:Geom_input)->tuple([NDArray[float],NDArray[float]]):  
    """
    Compute the top surface of a kinematic slab as a polyline starting from the trench.

    The slab surface is discretised into straight segments of length `g_input.sub_dl`.
    At each step we compute the local bending angle at the current and next arc-length
    positions, average them, and use that mean angle to advance to the next point.

    Angle convention
    ----------------
    theta is measured with respect to the positive horizontal x-axis.

             theta
    x-axis  -------\------
                    \ theta
                     \/

    Algorithm (summary)
    -------------------
    1. Initialise `top_slab` with the trench point.
    2. Initialise the arc-length `lgh = 0.0` (distance measured along the slab surface).
    3. While the current point is above the model bottom boundary (`y > ymin`):
       a. Set `lghn = lgh + g_input.sub_dl`.
       b. Compute `theta  = theta(lgh)`  and `theta1 = theta(lghn)`.
       c. Compute `theta_mean = 0.5 * (theta + theta1)`.
       d. Use `theta_mean` to advance one step and append the new point to `top_slab`.
       e. Update `lgh = lghn`.

    Returns
    -------
    top_slab : (n_segment, 2) ndarray
        Coordinates (x, y) of the slab top surface polyline.
    theta_mean : float
        Mean bending angle used for the last segment (or an average over segments,
        depending on your implementation).

    Raises
    ------
    ValueError
        If the selected slab-surface method is not implemented (only "custom" is
        currently supported).
    """

 
    if g_input.sub_type == 'Costum':
        # Initialise the theta_mean and slab_top array
        theta_mean = np.zeros([2],dtype=float)
        slab_top   = np.zeros([2,2],dtype=float)
        
        # compute the bending angle as a function of L
        slab_top[0,0] = g_input.sub_trench[0]
        slab_top[0,1] = 0.0

        lgh = 0.0
        lghn = 0.0

        dl = g_input.sub_dl

        it = 0 

        statement = True 
        while statement:
            lghn += dl
            theta1 = compute_bending_angle(g_input,lgh) # bending angle at the beginning of the segment
            theta2 = compute_bending_angle(g_input,lghn) # bending angle at the end of the segment
            theta = 0.5*(theta1+theta2) # mean bending angle
            theta_mean[it]= theta
            theta_meani_1 = theta
            # Find the middle of the slab
            slab_topi_ix = slab_top[it,0]+dl*(np.cos(theta)) # middle of the slab at the end of the segment x
            slab_topi_iz = slab_top[it,1]-dl*(np.sin(theta)) # middle of the slab at the end of the segment z


            if it+1 > len(slab_top[:,0])-1:
                slab_top = np.vstack([slab_top,[slab_topi_ix,slab_topi_iz]])
                theta_mean = np.append(theta_mean,theta_meani_1)
            else: 
                slab_top[it+1,0] = slab_topi_ix
                slab_top[it+1,1] = slab_topi_iz
                theta_mean[it] = theta
                theta_mean[it+1] = theta
            if g_input.y[0] != -1e23: 
                x = slab_top[it+1,1]
                statement = x > g_input.y[0]
                if not statement:
                    dz = slab_top[it,1] - g_input.y[0]
                    dx = dz / np.tan(theta)
                    slab_top[it+1,0] = slab_top[it,0] + dx
                    slab_top[it+1,1] = g_input.y[0]
            it = it+1
            lgh = lghn
            
    else: 
        raise ValueError("There is not yet any alternative to costum subduction yet, please, be patient")
        
    return slab_top, theta_mean

#------------------------------------------------------------------------------------------------------------
def compute_bending_angle(g_input:Geom_input
                        ,lgh: float):
    
    """compute_bending_angle: 
    inputs: 
    geometrical information
    current length of the slab 
    Returns:
        theta:float -> for a given bending angle function, return theta = f(l) -> l the actual distance along the slab surface from the trench
    """
    if g_input.sub_constant_flag:
        theta = g_input.sub_theta_max
    else:
        if lgh > g_input.sub_Lb:
            theta = g_input.sub_theta_max
        else:
            theta = ribe_angle(g_input.sub_theta_max, g_input.sub_Lb, lgh)
            if theta<g_input.sub_theta_0: 
                theta = g_input.sub_theta_0
    
    return theta

#--------------------------------------------------------------------------------------------------------------
def ribe_angle(theta_max: float
                ,Lb: float
                ,lgh: float) -> float:

    """ribe_angle 
    inputs: 
    theta_max : float -> maximum angle of the slab
    Lb : float -> critical along slab distance where the bending is occuring 
    lgh : float -> current position along the slab surface  

    Returns:
        theta :float -> Bending angle at the local point along the top surface slab. 
    """

    theta = theta_max*lgh**2*(3*Lb-2*lgh)/(Lb**3)

    return theta

def function_create_subducting_plate_geometry(g_input:Geom_input,
                                 )-> tuple[NDArray[np.float64],NDArray[np.float64],NDArray[np.float64],NDArray[np.float64],NDArray[np.float64] | None,NDArray[np.float64] | None, Geom_input]: 
    """
    Create the main subducting-plate geometry from input parameters.

    This routine builds the coordinate arrays describing the slab top and bottom
    surfaces and, if requested/available, the oceanic Moho. The returned arrays are
    used to define the Gmsh geometry and subsequent mesh generation. The input
    geometry object may be updated (e.g., derived quantities, validated parameters,
    or cached geometry).

    Parameters
    ----------
    g_input : Geom_input
        Object containing all input geometric parameters required to construct the
        slab and associated interfaces.

    Returns
    -------
    ax : np.ndarray
        x-coordinates of the slab top surface.
    ay : np.ndarray
        y-coordinates of the slab top surface.
    bx : np.ndarray
        x-coordinates of the slab bottom surface.
    by : np.ndarray
        y-coordinates of the slab bottom surface.
    ox : np.ndarray | None
        x-coordinates of the oceanic Moho. Can be `None`/empty if the crust is not
        defined or not requested.
    oy : np.ndarray | None
        y-coordinates of the oceanic Moho. Can be `None`/empty if the crust is not
        defined or not requested.
    g_input : Geom_input
        Updated geometry input object (may include derived or validated fields).
    """
    
    # Prepare the slab surface. Slab surface can be derived from real data, or just created ad hoc by the slab routines
    # Add the extra node that is required for the interface with the lithosphere. 
    slab_top, theta_mean = find_slab_surface(g_input)

    ax = slab_top[:,0]
    ay = slab_top[:,1] 
        
    # Create the channel using the subduction interface as guide
    #cx,cy = function_create_subduction_channel(ax,ay,theta_mean,g_input)
    if g_input.ocr != 0.0:
        ox,oy = generate_parallel_layer_subducting_plate(ax,ay,theta_mean,g_input.ocr)
    else: 
        ox = None
        oy = None 
    # Correct the slab surface and find the extra node
    ax,ay,theta_mean = find_extra_node(ax,ay,theta_mean,g_input)
    # Correct the subduction channel as well finding the extranode 
    # I needed to correct first the slab and then the channel, because otherwise it was
    # creating an unrealistic bending of the weak zone. Initially I was recomputing the angle 
    # between the linear segment, using an average. It is more convinient use two // lines and then 
    # correcting them. 
    bx,by = generate_parallel_layer_subducting_plate(ax,ay,theta_mean,g_input.slab_tk)

    # update the g_input
    g_input.theta_in_slab = theta_mean[0]


    return ax,ay,bx,by,ox,oy,g_input

#---------------------------------------------------------------------------------------------
def generate_parallel_layer_subducting_plate(sx:NDArray[np.float64],
                                      sy:NDArray[np.float64],
                                      th:NDArray[np.float64],
                                      lt:float)->tuple[NDArray[np.float64],NDArray[np.float64]]:
    """
    Compute the coordinates of an internal layer surface within the subducting plate.
    
    Given the slab top surface coordinates and a local slab bending angle, this
    function constructs a parallel/offset surface at a distance `lt` (e.g., oceanic
    crust thickness or slab thickness), representing an internal layer boundary
    within the subducting plate.
    
    Parameters
    ----------
    sx : NDArray[np.float64]
        x-coordinates of the slab top surface.
    sy : NDArray[np.float64]
        y-coordinates of the slab top surface.
    th : float
        Local slab bending angle associated with the surface points (degrees or
        radians depending on the implementation; must be consistent with the
        trigonometric functions used).
    lt : float
        Layer thickness used to offset the surface (e.g., oceanic crust thickness
        or slab thickness) (SI units: [m]).
    
    Returns
    -------
    cx : NDArray[np.float64]
        x-coordinates of the layer-defining surface.
    cy : NDArray[np.float64]
        y-coordinates of the layer-defining surface.
    """

    cx = np.zeros([np.amax(sx.shape),1])
    cy = np.zeros([np.amax(sx.shape),1])
    # Loop over the interface of the slab and find the points on the top of the surface of the subduction channel: the point on the of the top of the channel are perpendicular to the slab interface#
    # Compute the top surface of the subduction channel
    cx = sx - lt*np.sin(th)
    cy = sy - lt*np.cos(th)

    e_node2,t_ex1 = _find_e_node(cx,cy,cx*0.0,-np.min(sy),flag=False)
    cx,cy,t  = _correct_(cx,cy,e_node2,-np.min(sy),cy*0.0,0.0)

    cx_n = cx[(cx>=0.0) & (cy>=np.min(sy))]
    cy_n = cy[(cx>=0.0) & (cy>=np.min(sy))]

    return cx_n,cy_n
    
#-----------------------------------------------------------------------------------------------------------------
  
def _find_e_node(ax:NDArray[np.float64],ay:NDArray[np.float64],t:NDArray[np.float64],lt:float,flag=False):
    if lt == 0.0 and flag == False: 
    
        return [],[]
    
    for i in range (0,len(ax)-1):
    
        if flag == False: 
            a = i 
            b = i+1 
        else: 
            a = i+1 
            b = i 
    
        if ay[i] == -lt:
            e_node = 0
    
        elif ay[a] > -lt and ay[b] < -lt:
            e_node = 1
            index_extra_node = i+1
            t_ex = t[i]/2+t[i+1]/2
            break
    if e_node == 0:
        return [],[]
    else:
        return index_extra_node,t_ex

#-----------------------------------------------------------------------------------------------------------------

def _correct_(ax:NDArray[np.float64],
              ay:NDArray[np.float64],
              index_extra_node:int,
              lt:float,
              t:NDArray[np.float64],
              tex:float)->tuple[NDArray[np.float64],NDArray[np.float64],NDArray[np.float64]]:
    
    if index_extra_node == []:
        return ax,ay,t
    
    ax_ex = ax[index_extra_node-1] + ( (ax[index_extra_node] - ax[index_extra_node-1])
                 * (lt - np.abs(ay[index_extra_node-1])) ) / (np.abs(ay[index_extra_node]) - np.abs(ay[index_extra_node-1]))
    
    ax = np.insert(ax,index_extra_node,ax_ex) #ax[index_extra_node-2] + ( (ax[index_extra_node] - ax[index_extra_node-2])* (lt - np.abs(ay[index_extra_node-2])) ) / (np.abs(ay[index_extra_node]) - np.abs(ay[index_extra_node-2])))
    
    ay = np.insert(ay,index_extra_node,-lt)  
    
    t = np.insert(t,index_extra_node,tex)  
    
    return ax,ay,t 
    
#-----------------------------------------------------------------------------------------------------------------

def find_extra_node(ax:NDArray[np.float64],
                    ay:NDArray[np.float64],
                    t:NDArray[np.float64],
                    g_input:Geom_input)->tuple[NDArray[np.float64],NDArray[np.float64],NDArray[np.float64]]:

    #-- Find nodes 
    e_node_uc,t_ex1 = _find_e_node(ax,ay,t,g_input.cr*(1-g_input.lc))
    ax,ay,t = _correct_(ax,ay,e_node_uc,g_input.cr*(1-g_input.lc),t,t_ex1)

    
    e_node_lc,t_ex2 = _find_e_node(ax,ay,t,g_input.cr)
    ax,ay,t = _correct_(ax,ay,e_node_lc,g_input.cr,t,t_ex2)

    
    e_node_lit,t_ex3 = _find_e_node(ax,ay,t,g_input.ns_depth)
    ax,ay,t = _correct_(ax,ay,e_node_lit,g_input.ns_depth,t,t_ex3)

        
    e_node_lit,t_ex3 = _find_e_node(ax,ay,t,g_input.decoupling)
    ax,ay,t = _correct_(ax,ay,e_node_lit,g_input.decoupling,t,t_ex3)


    return ax,ay,t

#----------------------------------------------------------------------------------------------------------------------------------

def _create_points(mesh:gmsh.model,                                            # I am not able to classify 
                   x:float,                                         # coordinate point/points
                   y:float,                                         # coordinate point/points
                   res:float,                                       # resolution of the ppint
                   tag_pr:int,                                      # maximum tag of the previous group
                   point_flag:bool=False)-> tuple[int, int, float]: # a flag to highlight that there is only one point, yes, lame. 
    
    """
    Create Gmsh point(s) and keep track of their tags and coordinates.

    This helper wraps the Gmsh point-creation call to support creating one or many
    points, while explicitly tracking the assigned tags so they can be referenced
    consistently later when building lines/surfaces.

    Parameters
    ----------
    mesh : gmsh.model
        the mesh model 
    x : float | np.ndarray
        x-coordinate(s) of the point(s) [m]. Can be a scalar or an array.
    y : float | np.ndarray
        y-coordinate(s) of the point(s) [m]. Can be a scalar or an array.
    res : float
        Target mesh size (characteristic length) assigned to the point(s).
    tag_pr : int
        Previous/starting tag used to track point IDs across calls. Since Gmsh
        can assign tags internally, this value is used to keep bookkeeping
        consistent.

    Returns
    -------
    max_tag : int
        Maximum tag assigned during this function call.
    tag_list : list[int]
        List of tags for the created point(s), ordered consistently with the input
        coordinates.
    coord : np.ndarray
        Coordinates of the created point(s), typically shaped as (N, 2) for (x, y).
        If useful downstream, you may include tags alongside coordinates (e.g.,
        (N, 3) with [tag, x, y])—in that case, document the exact convention.

    Notes
    -----
    This function solves the practical problem of creating a set of points (1..N)
    in Gmsh while retaining a reliable mapping between coordinates and point tags,
    so that subsequent geometry construction can reference points coherently.
    """

    tag_list = []

    if point_flag == True:
        coord = np.zeros([3,1],dtype=np.float64)
        mesh.geo.addPoint(x, y, 0.0,res,tag_pr+1) 
        tag_list.append(tag_pr+1)
        coord[0] = x 
        coord[1] = y
        coord[2] = tag_pr+1
        
    else:
        coord = np.zeros([3,len(x)],dtype=np.float64)
        for i in range(len(x)):

            mesh.geo.addPoint(x[i], y[i], 0.0, res, tag_pr + 1 + i) 
            tag_list.append(tag_pr+1+i)
            coord[0,i]   = x[i]
            coord[1,i]   = y[i] 
            coord[2,i]   = tag_pr + 1 + i
            
    
    max_tag = np.max(tag_list)

    
    return max_tag,tag_list,coord,mesh

#------------------------------------------------------------------------------------------------------------------------------

def _create_lines(mesh_model:gmsh.model,
                  previous:int,
                  tag_p:list,
                  flag=False)-> tuple[int, list, NDArray[np.int32], gmsh.model]:

    """
    Create Gmsh line(s) from point tags and keep track of their tags and connectivity.
    
    This helper wraps the Gmsh line-creation call to support creating one or many
    lines, while explicitly tracking the assigned line tags so they can be
    referenced consistently later when building curve loops, surfaces, and physical
    groups.
    
    Parameters
    ----------
    mesh : gmsh.model
        Gmsh model object to be updated.
    previous : int
        Previous/starting line tag used for bookkeeping across calls.
    tag_p : list[int]
        Point tags used to create the lines. Typically an ordered list of point
        tags where consecutive pairs define segments.
    
    Returns
    -------
    max_tag : int
        Maximum line tag assigned during this function call.
    tag_l : list[int]
        List of tags for the created line(s), ordered consistently with the created
        segments.
    lines : np.ndarray
        Array containing line metadata, typically shaped (N, 3) with rows
        `[p1, p2, tag]`, where `p1` and `p2` are point tags and `tag` is the line tag.
    mesh_model : gmsh.model
        Updated Gmsh model containing the created line(s).
    
    Notes
    -----
    This function solves the practical problem of generating a set of lines from an
    ordered list of point tags while keeping a reliable mapping between endpoint
    tags and the created line tags for downstream geometry construction.
    """
    
    
    len_p = len(tag_p)-1
    tag_l = []
    previous = previous+1
    lines = np.zeros([3,len_p],dtype=np.int32)
    for i in range(len_p): 
        a = tag_p[i]
        b = tag_p[i+1]
        'it appears that the gmsh is an horrible person: i try to give the golden rule that the tag of the points are following an order,'
        'which entails that the minimum of the two point defining a segment is the first -> such that it is always possible determine the  order'
 
        tag_1 = np.min([a,b])
        tag_2 = np.max([a,b]) 
        mesh_model.geo.addLine(tag_1,tag_2,previous+i)
        tag_l.append(previous+i)
        lines[0,i] = tag_1 
        lines[1,i] = tag_2 
        lines[2,i] = previous+i
        
    max_tag = np.max(tag_l)
    
    return max_tag,tag_l,lines,mesh_model
#------------------------------------------------------------------------------------------------------------
def assign_phases(dict_surf:dict, 
                  cell_tags:int,
                  phase:dolfinx.fem.Function)->dolfinx.fem.Function:
    """Assigns phase tags to the mesh based on the provided surface tags."""
    for tag, value in dict_surf.items():
        indices = cell_tags.find(value) 
        phase.x.array[indices] = np.full_like(indices,  value , dtype=PETSc.IntType)
    
    phase.x.scatter_forward()
    
    return phase 
#---------------------------------------------------------------------------------------------------------------
# Break this glass in case of needs. 
# def debug_plot(target,global_line,global_point,color):
#    for i in range(len(target)):
#        line = np.abs(target[i])
#        
#        p0   = global_line[0,line-1]
#        p1   = global_line[1,line-1]
#        coord_x = [global_point[0,p0-1],global_point[0,p1-1]]            
#        coord_y = [global_point[1,p0-1],global_point[1,p1-1]]