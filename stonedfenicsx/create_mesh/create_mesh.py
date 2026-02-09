# input for iFieldstone 

from stonedfenicsx.package_import import *

from stonedfenicsx.numerical_control import IOControls, NumericalControls
from stonedfenicsx.scal              import _scaling_mesh,Scal
from stonedfenicsx.utils             import print_ph
from dolfinx.mesh       import create_submesh
from .aux_create_mesh   import Mesh, Domain, Class_Points, Class_Line, Geom_input, dict_tag_lines, dict_surf,find_line_index,create_loop,function_create_subducting_plate_geometry
from .aux_create_mesh import assign_phases,from_line_to_point_coordinate

#------------------------------------------------------------------------------------------------------
def create_mesh(ioctrl:IOControls
                ,sc:Scal
                ,g_input:Geom_input
                ,ctrl:NumericalControls)->Mesh:
    """
    Create a Gmsh model and convert it into a FEniCSx mesh.

    This function generates the computational geometry using Gmsh, builds the
    corresponding `.msh` file, and imports it into dolfinx as a finite element mesh.

    Parameters
    ----------
    ioctrl : IOControls
        I/O controller handling file paths and output directories.
    sc : Scal
        Scaling object storing nondimensionalisation parameters.
    g_input : Geom_input
        Geometrical input defining the domain and mesh construction.
    ctrl : NumericalControls
        Numerical controls determining the problem setup.

    Returns
    -------
    Mesh
        Mesh wrapper object containing the global mesh, extracted subdomains,
        and associated boundary/cell tags.
    """

    # Collect the rank and comm 
    rank = MPI.COMM_WORLD.Get_rank()  # 0, 1, ..., size-1
    # Perform the create mesh routines in the rank 0
    if rank == 0: 
        g_input = create_gmesh(ioctrl,g_input,ctrl)
    # Convert the mesh from gmsh to mesh objects 
    M = create_mesh_object(sc,ioctrl, g_input)
    
    return M
#-----------------------------------------------------------------------------------------------
def create_gmesh(ioctrl   : IOControls,
                 g_input  : Geom_input,
                 ctrl     : NumericalControls):
    """
    Create a Gmsh geometry model from the provided geometrical input.

    This function generates a `.msh` file, which is later imported and converted
    into a finite element mesh compatible with FEniCSx (dolfinx).

    The function also updates the `Geom_input` object by computing and storing
    the initial slab bending angle and the outgoing angle of the subducting plate.
    This information is required to construct the left thermal boundary conditions.

    The lateral extent of the global domain depends on the slab geometry and the
    maximum model depth. Therefore, the function modifies the domain geometry by
    extending the horizontal size by an additional 60 km from the intersection
    between the slab top surface and the maximum depth of the model.

    Parameters
    ----------
    ctrl : NumericalControls
        Numerical controls defining the simulation type.
    ctrlio : IOControls
        I/O controller storing input/output directories.
    g_input : Geom_input
        Geometrical input parameters used to construct the domain.

    Returns
    -------
    Geom_input
        Updated geometrical input object containing the computed slab angles and
        modified domain extent.
    """


    min_x           = g_input.x[0]        # The beginning of the model is the trench of the slab
    max_x           = g_input.x[1]                 # Max domain x direction
    max_y           = g_input.y[1] 
    min_y           = g_input.y[0]                # Min domain y direction
    # Set up slab top surface a
    van_keken =ctrl.van_keken
    if van_keken == 1: 
        min_x           =0.0 # The beginning of the model is the trench of the slab
        max_x           = 660e3                 # Max domain x direction
        max_y           = 0.0
        min_y           = -600.0e3               # Min domain y direction            
        g_input.x[0]   = min_x
        g_input.x[1]   = max_x
        g_input.y[1]   = max_y
        g_input.y[0]   = min_y
    
    # Create the subducting plate main geometrical point and lines using the information of g_input
    (slab_x
     ,slab_y
     ,bot_x
     ,bot_y
     ,oc_cx
     ,oc_cy
     ,g_input) = function_create_subducting_plate_geometry(g_input)
    
    if slab_x[-1]>g_input.x[1]:
        print_ph('Shortcoming: the slab is out of the domain, please increase the domain size, I add 60 km to the domain along x direction, fear not')
        g_input.x[1] = slab_x[-1]+60e3
        max_x        = g_input.x[1]
        
    min_x           = g_input.x[0] # The beginning of the model is the trench of the slab
    max_x           = g_input.x[1]          
        
    if g_input.cr != 0:
        ind_oc_cr = np.where(slab_y == -g_input.cr)[0][0]
        if g_input.lc !=0:     
            ind_oc_cr = np.where(slab_y == -g_input.cr)[0][0]
        else: 
            ind_oc_lc = []
    else: 
        ind_oc_cr = []
        ind_oc_lc = []
        
    ind_oc_lt = np.where(slab_y == -g_input.ns_depth)[0][0]
    
    
    mesh_model = create_gmsh(slab_x
                             ,slab_y
                             ,bot_x
                             ,bot_y
                             ,oc_cx
                             ,oc_cy
                             ,g_input) 

    theta = np.arctan2((slab_y[-1]-slab_y[-2]),(slab_x[-1]-slab_x[-2]))
    
    g_input.theta_out_slab = theta   # Convert to degrees

    mesh_model.geo.removeAllDuplicates()

    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay

    gmsh.option.setNumber("Mesh.Optimize", 1)

    mesh_model.mesh.generate(2)
    
    mesh_name = os.path.join(ioctrl.path_save,ioctrl.sname)
    
    gmsh.write("%s.msh"%mesh_name)
    
    gmsh.finalize()
    
    return g_input
#--------------------------------------------------------------------------------------------------------
def create_domain_A(mesh_model:gmsh.model
                    ,CP:Class_Points
                    ,LC:Class_Line
                    ,g_input:Geom_input)->gmsh.model:
    """
    Create the subducting-plate loop in the Gmsh model.
    
    Parameters
    ----------
    mesh_model : gmsh.model
        Gmsh model object containing the geometry/mesh entities.
    CP : Class_Points
        Container of points defining the global mesh (coordinates, point IDs, target resolution).
    LP : Class_Line
        Container of lines defining the global mesh (line IDs and point connectivity).
    g_input : Geom_input
        Geometry input parameters.
    
    Returns
    -------
    gmsh.model
        The updated Gmsh model with the sub-domain loop entities added.
    """



    if g_input.ocr != 0.0:
        
        l_list     = [LC.lines_oc[2,:],-LC.lines_B[2,-1],-LC.lines_BS[2,::-1], LC.lines_L[2,0]]
        mesh_model = create_loop(l_list, mesh_model, 10)

        # Oceanic crust  
        l_list     = [LC.lines_S[2,:], LC.lines_B[2,1], -LC.lines_oc[2,::-1],  -LC.lines_L[2,-1]]
        mesh_model = create_loop(l_list, mesh_model, 15)
    else:
        l_list     = [LC.lines_S[2,:],LC.lines_B[2,-1],-LC.lines_BS[2,::-1], -LC.lines_L[2,0]]  
        mesh_model = create_loop(l_list, mesh_model, 15)
        
        
    print('Finished to generate the curved loop for domain A [Subducting plate]')

    return mesh_model

#--------------------------------------------------------------------------------------------------------
def create_domain_B(mesh_model
                    ,CP
                    ,LC
                    ,g_input):

    """
    Create the Wedge loop in the Gmsh model.
    
    Parameters
    ----------
    mesh_model : gmsh.model
        Gmsh model object containing the geometry/mesh entities.
    CP : Class_Points
        Container of points defining the global mesh (coordinates, point IDs, target resolution).
    LP : Class_Line
        Container of lines defining the global mesh (line IDs and point connectivity).
    g_input : Geom_input
        Geometry input parameters.
    
    Returns
    -------
    gmsh.model
        The updated Gmsh model with the sub-domain loop entities added.
    """


    index = find_line_index(LC.lines_S,CP.coord_sub,g_input.ns_depth)
    index = index 
    buf_array = LC.lines_S[2,index:]
    buf_array = -buf_array 
    buf_array = buf_array[::-1]


    
    l_list     = [-LC.lines_R[2,-1],-LC.lines_B[2,0],buf_array, LC.lines_L_ov[2,:]]
    mesh_model = create_loop(l_list, mesh_model, 20)
    

    
    print('Finished to generate the curved loop for domain B [Wedge]')
    return mesh_model 
#-------------------------------------------------------------------------------------------------------------
def create_domain_C(mesh_model
                    ,CP
                    ,LC
                    ,g_input):


    """
    Create the Overriding plate loop in the Gmsh model.
    
    Parameters
    ----------
    mesh_model : gmsh.model
        Gmsh model object containing the geometry/mesh entities.
    CP : Class_Points
        Container of points defining the global mesh (coordinates, point IDs, target resolution).
    LP : Class_Line
        Container of lines defining the global mesh (line IDs and point connectivity).
    g_input : Geom_input
        Geometry input parameters.
    
    Returns
    -------
    gmsh.model
        The updated Gmsh model with the sub-domain loop entities added.
    """

    if g_input.cr !=0:
            
        if g_input.lc !=0:
            
            index_a    = find_line_index(LC.lines_S,CP.coord_sub,g_input.cr)
            index_b    = find_line_index(LC.lines_S,CP.coord_sub,g_input.ns_depth)-1
            buf_array = LC.lines_S[2,index_a:index_b+1]
            buf_array = -buf_array[::-1]
        
            l_list     = [-LC.lines_R[2,2],-LC.lines_L_ov[2,:],buf_array,LC.lines_cr[2,:]]
            mesh_model = create_loop(l_list, mesh_model, 25)
            
            index_a    = find_line_index(LC.lines_S,CP.coord_sub,(1-g_input.lc)*g_input.cr)
            index_b    = find_line_index(LC.lines_S,CP.coord_sub,g_input.cr)-1

            buf_array = -LC.lines_S[2,index_a:index_b+1]
            buf_array = buf_array[::-1]
            
            l_list = [-LC.lines_R[2,1],-LC.lines_cr[2,:],buf_array,LC.lines_lcr[2,:]]
            mesh_model = create_loop(l_list, mesh_model, 35)


            index_a    = 0
            index_b    = find_line_index(LC.lines_S, CP.coord_sub,(1-g_input.lc)*g_input.cr)-1
            buf_array = -LC.lines_S[2,index_a:index_b+1]
            buf_array = buf_array[::-1]

            l_list = [LC.lines_R[2,0],-LC.lines_lcr[2,:],buf_array,LC.lines_T[2,:]]
            mesh_model = create_loop(l_list, mesh_model, 30)
        else:
            index_a    = find_line_index(LC.lines_S,CP.coord_sub,g_input.cr)
            index_b    = find_line_index(LC.lines_S,CP.coord_sub,g_input.ns_depth)-1
            buf_array = LC.lines_S[2,index_a:index_b+1]
            buf_array = -buf_array[::-1]
        
            l_list     = [-LC.lines_R[2,1],-LC.lines_L_ov[2,:],buf_array,LC.lines_cr[2,:]]
            mesh_model = create_loop(l_list, mesh_model, 25)
            
            index_a    = 0
            index_b    = find_line_index(LC.lines_S, CP.coord_sub,g_input.cr)-1
            buf_array = -LC.lines_S[2,index_a:index_b+1]
            buf_array = buf_array[::-1]

            l_list = [LC.lines_R[2,0],-LC.lines_cr[2,:],buf_array,LC.lines_T[2,:]]
            mesh_model = create_loop(l_list, mesh_model, 30)
    else:   
        index    = find_line_index(LC.lines_S,CP.coord_sub,g_input.ns_depth)-1
        buf_array = LC.lines_S[2,0:index+1]
        buf_array = -buf_array[::-1]
        
        l_list     = [-LC.lines_R[2,0],-LC.lines_L_ov[2,:],buf_array,LC.lines_T[2,:]]
        mesh_model = create_loop(l_list, mesh_model, 25)

    print('Finished to generate the curved loop for domain C [Crust]')
    return mesh_model

#--------------------------------------------------------------------------------------------------------
def create_physical_line(CP:Class_Points
                         ,LC:Class_Line
                         ,g_input:Geom_input
                         ,mesh_model:gmsh.model)->gmsh.model:

    """Create the physical line using the lines.
    
    Parameters
    ----------
    mesh_model : gmsh.model
        Gmsh model object containing the geometry/mesh entities.
    CP : Class_Points
        Container of points defining the global mesh (coordinates, point IDs, target resolution).
    LP : Class_Line
        Container of lines defining the global mesh (line IDs and point connectivity).
    g_input : Geom_input
        Geometry input parameters.
    
    Returns
    -------
    gmsh.model
        The updated Gmsh model with the physical line loop entities added.
    """

    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}
    
    mesh_model.addPhysicalGroup(1, LC.tag_L_T, tag=dict_tag_lines['Top'])
    
    # Find point above the lithosphere 
    for i in range(len(LC.tag_L_R)):
        L = LC.tag_L_R[i]
        p0,p1,cx,cy = from_line_to_point_coordinate(L,LC.line_global, CP.global_points)
        if cy[0] == -g_input.ns_depth or cy[1] == -g_input.ns_depth:
            break 
    mesh_model.addPhysicalGroup(1, LC.tag_L_R[0:i+1], tag=dict_tag_lines['Right_lit'])

    mesh_model.addPhysicalGroup(1, LC.tag_L_R[i+1:], tag=dict_tag_lines['Right_wed'])

    mesh_model.addPhysicalGroup(1, [LC.tag_L_B[0]], tag=dict_tag_lines['Bottom_wed'])

    mesh_model.addPhysicalGroup(1, LC.tag_L_B[1:], tag=dict_tag_lines['Bottom_sla'])
    
    mesh_model.addPhysicalGroup(1, LC.tag_L_Bsub, tag=dict_tag_lines['Subduction_bot'])

    mesh_model.addPhysicalGroup(1, LC.tag_L_L, tag=dict_tag_lines['Left_inlet'])

    for i in range(len(LC.tag_L_sub)):
        L = LC.tag_L_sub[i]
        p0,p1,cx,cy = from_line_to_point_coordinate(L,LC.line_global, CP.global_points)
        if cy[0] == -g_input.ns_depth or cy[1] == -g_input.ns_depth:
            break 
    
    mesh_model.addPhysicalGroup(1, LC.tag_L_sub[0:i+1],   tag=dict_tag_lines['Subduction_top_lit'])

    mesh_model.addPhysicalGroup(1, LC.tag_L_sub[i:],   tag=dict_tag_lines['Subduction_top_wed'])

    if g_input.ocr != 0.0:
        mesh_model.addPhysicalGroup(1, LC.tag_L_oc,    tag=dict_tag_lines['Oceanic'])
  
        mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}


    mesh_model.addPhysicalGroup(1, LC.tag_L_ov,    tag=dict_tag_lines['Overriding_mantle'])

    if g_input.cr !=0: 

        mesh_model.addPhysicalGroup(1,    LC.tag_L_cr, tag=dict_tag_lines['Crust_overplate'])

        if g_input.lc !=0:
            mesh_model.addPhysicalGroup(1, LC.tag_L_Lcr,tag=dict_tag_lines['LCrust_overplate'])
    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    return mesh_model 
#----------------------------------------------------------------------------------------------------------------------
def create_gmsh(sx:np.ndarray,      # subduction x
                sy:np.ndarray,        # subdcution y 
                bsx:np.ndarray,       # bottom subduction x
                bsy:np.ndarray,       # bottom subduction y 
                oc_cx:np.ndarray,     # oceanic cx 
                oc_cy:np.ndarray,     # oceanic cu
                g_input:Geom_input)->gmsh.model:  # geometry input class 
    """
    Create a Gmsh model from the slab and crust geometry.

    Parameters
    ----------
    sx : np.ndarray
        x-coordinates of the top surface of the subducting plate.
    sy : np.ndarray
        y-coordinates of the top surface of the subducting plate.
    bsx : np.ndarray
        x-coordinates of the bottom surface of the subducting plate.
    bsy : np.ndarray
        y-coordinates of the bottom surface of the subducting plate.
    oc_cx : np.ndarray
        x-coordinates of the oceanic crust Moho.
    oc_cy : np.ndarray
        y-coordinates of the oceanic crust Moho.
    g_input : Geom_input
        Object containing the geometrical input parameters used to construct the model.

    Returns
    -------
    object
        The generated Gmsh model handle (`gmsh.model`), containing the geometry
        definition (points, curves, surfaces) and the associated physical groups
        (mesh tags).
    """
    
    # -> USE GMSH FUNCTION 
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    mesh_model = gmsh.model()
    # Create the point class containing the physical point of the mesh
    CP = Class_Points()
    mesh_model = CP.update_points(mesh_model,sx,sy,bsx,bsy,oc_cx,oc_cy,g_input)
    # Create the line class containing the lines of the mesh 
    LC = Class_Line()
    mesh_model = LC.update_lines(mesh_model, CP, g_input)
    # Create the physical lines 
    mesh_model = create_physical_line(CP,LC,g_input,mesh_model)
    # Create the sub-domains of the mesh
    mesh_model = create_domain_A(mesh_model, CP, LC, g_input)
    mesh_model = create_domain_B(mesh_model, CP, LC, g_input)
    mesh_model = create_domain_C(mesh_model, CP, LC, g_input)    
    # Create the  surface 
    if g_input.ocr != 0.0:
        Left_side_of_subduction_surf   = gmsh.model.geo.addPlaneSurface([10],tag=100) # Left side of the subudction zone
        Oceanic_Crust_surf             = gmsh.model.geo.addPlaneSurface([15],tag=150) # Left side of the subudction zone
    else:
        Left_side_of_subduction_surf   = gmsh.model.geo.addPlaneSurface([15],tag=150) # Left side of the subudction zone
    Wedge                          = gmsh.model.geo.addPlaneSurface([20],tag=200) # Right side of the subudction zone    
    Lithhospheric_Mantle_surf      = gmsh.model.geo.addPlaneSurface([25],tag=250) # Right mantle
    if g_input.cr !=0:
        Crust_UC_surf                  = gmsh.model.geo.addPlaneSurface([30],tag=300) # Crust LC
        if g_input.lc !=0:
            Crust_LC_surf                  = gmsh.model.geo.addPlaneSurface([35],tag=350) # Crust LC
    

    # Create the physical surface the effective domain
    mesh_model.geo.synchronize()

    mesh_model.addPhysicalGroup(2, [Left_side_of_subduction_surf],  tag=dict_surf['sub_plate'])
    if g_input.ocr != 0.0:
        mesh_model.addPhysicalGroup(2, [Oceanic_Crust_surf],            tag=dict_surf['oceanic_crust'])
    mesh_model.addPhysicalGroup(2, [Wedge],                         tag=dict_surf['wedge'])
    mesh_model.addPhysicalGroup(2, [Lithhospheric_Mantle_surf],     tag=dict_surf['overriding_lm'])
    if g_input.cr !=0:
        mesh_model.addPhysicalGroup(2, [Crust_UC_surf],                 tag=dict_surf['upper_crust'])
        if g_input.lc !=0:
            mesh_model.addPhysicalGroup(2, [Crust_LC_surf],                 tag=dict_surf['lower_crust'])


    return mesh_model 

#----------------------------------------------------------------------------------------------------------------------
def create_mesh_fenicsx(mesh:meshio._mesh.Mesh
                        ,cell_type:str 
                        ,prune_z:bool=False)->dolfinx.mesh.Mesh:

    """Convert gmsh into meshio object
    Parameters 
    -----------
        mesh : meshio._mesh.Mesh
            meshio object that contains the .msh model
        cell_type: str 
                element type to process {triangle} or {line} 
        prune_z : Bool 
            flag to remove the z coordinate from the mesh model
    Returns
    -----------
        _type_: _description_
    """

    # From the tutorials of dolfinx
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data.astype(np.int32)]})
    
    return out_mesh
#------------------------------------------------------------------------------------------------------------------
def extract_facet_boundary(Mesh:dolfinx.mesh.Mesh
                           ,Mfacet_tag:dolfinx.mesh.MeshTags
                           ,submesh:dolfinx.mesh.Mesh
                           ,sm_vertex_maps:np.ndarray
                           ,boundary:list
                           ,m_id:int)->tuple([np.ndarray,np.ndarray]):
    r"""
    Extract facet indices for a submesh boundary from global boundary markers.

    This function builds the facet-index array (and corresponding marker values) needed to
    define boundary conditions on a *submesh*, starting from boundary tags defined on the
    *global mesh*.

    Why this is needed:
    Some physical boundaries are represented by multiple tagged pieces on the global mesh
    (e.g. the “top surface of the slab” may be split into separate segments depending on
    which neighboring region it touches). When imposing a single boundary condition on a
    subdomain, you often need to *merge* several of these tagged pieces into one boundary
    set for that subdomain.

    Example (schematic):
        Overriding plate
           \
    S(1)    \
             \__________
    S(2)      \  Wedge
               \

    For the slab subdomain boundary, you may need to combine S(1) and S(2).

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The global mesh.
    mfacet_tag : dolfinx.mesh.meshtags.MeshTags
        Facet MeshTags on the global mesh (dimension = tdim - 1).
    submesh : dolfinx.mesh.Mesh
        The subdomain mesh.
    sm_vertex_maps : np.ndarray | list[int]
        Mapping from submesh vertices to parent (global) mesh vertices.
    boundary : Sequence[int]
        List of global boundary marker IDs to be combined for this submesh boundary.
    m_id : int
        Marker ID to assign to the extracted facets on the submesh.

    Returns
    -------
    chosen_facets : np.ndarray
        Indices of submesh facets that belong to the extracted (internal/external) boundary.
    values : np.ndarray
        Array of shape (len(chosen_facets),) filled with `m_id`.
    """


    # Extract facet from the parent mesh. Boundary -> list of marker of the boundary [i.e., {5,6}].     
    chosen_one = []
    for tag in boundary:
        chosen_one.extend(Mfacet_tag.find(tag))
        
    #--     
     # Parent Mesh -> extract information. 
     # First extract information for any particular boundary (also composite), then select the facet that are belonging to this particular boundary 
     # then extract the nodes id {the function that creates the submesh release the maps to the parent mesh, we can exploit it for generating a list of
     # facet}
    #--     
    
    # Remove the double entries []    
    chosen_one = np.unique(chosen_one)
    # Create the topology => from the set of facet, generate an object containing number of facet and the point/vertices that define each of them
    Mesh.topology.create_connectivity(Mfacet_tag.dim, 0)
    # Extract connectivity
    facet_to_vertex = Mesh.topology.connectivity(Mfacet_tag.dim, 0)
    # From facet number to node 
    vertex_indices = [] # Empty list
    for f in chosen_one:
        vertex_indices.extend(facet_to_vertex.links(f))

    # Remove double entries
    vertex_indices = np.unique(vertex_indices)
    # -- 
    # Sub mesh 
    # --
    # Extract the vertex from submesh 
    cell_dim = submesh.topology.dim
    cell_to_vertex = submesh.topology.connectivity(cell_dim, 0)

    # Extract the vertices
    v_ids = cell_to_vertex.links(0)

    # Extract the indices and then find the unique one 
    all_vertex_ids = []
    for c in range(submesh.topology.index_map(cell_dim).size_local):
        all_vertex_ids.extend(cell_to_vertex.links(c))

    unique_vertex_ids = np.unique(all_vertex_ids)

    # Loop over the indeces of the nodes belonging to the given boundary from the parent mesh
    # -> find the index in the local sub-mesh mapping 
    # -> Collect the index from the local_mesh index 
    ind_facet = []
    for i in range(len(vertex_indices)): 
        ind = np.where(sm_vertex_maps == vertex_indices[i])[0]
        ind_facet.extend(unique_vertex_ids[ind])
    

    # 3. Loop over all the facet -> extract the node ids of the facet of the submesh if both of them exist in the array ind_facet
    # -> collect the number of the facet and collect into a list -> then turn into an array -> you have your boundary 
    connectivity = submesh.topology.connectivity(1, 0)   
    chosen_facet = []        
    for facet_index in range(submesh.topology.index_map(1).size_local):
        sub_mesh_vertex_index = connectivity.links(facet_index)
        if (np.isin(sub_mesh_vertex_index[0],ind_facet)) and (np.isin(sub_mesh_vertex_index[1],ind_facet)): 
            chosen_facet.append(facet_index)
    
    chosen_facet = np.asarray(chosen_facet,dtype=np.int32)
    values       = np.full(chosen_facet.size,m_id,dtype = np.int32)
    

    return  chosen_facet, values

#--------------------------------------------------------------------------------------------------------
def create_subdomain(mesh:dolfinx.mesh.Mesh
                     ,mesh_tag:dolfinx.mesh.MeshTags
                     ,facet_tag:dolfinx.mesh.MeshTags
                     ,phase_set:list
                     ,name:str
                     ,phase:dolfinx.fem.function.Function)->Domain:
    """Create the subdomain from the global mesh, and interpolate the phases from the global mesh to the local mesh

    Parameters
    ----------
        mesh : dolfinx.mesh.Mesh
            global mesh information
        mesh_tag : dolfinx.mesh.MeshTags
            the mesh tag of the surface {i.e., phase}
        facet_tag : dolfinx.mesh.MeshTags
            the mesh tag of the linear feature (e.g., subducting top surface)
        phase_set :list 
            the cell marker that constitute the subdomain
        name : str
            the name of the subdomain
        phase : dolfinx.fem.function.Function
            function that stores the information of the phase. 

    Returns
    ----------
        domain : Domain
              Class that contains the information of the subdomain. 
    """

    from dolfinx.mesh import meshtags
    #--------------------------------------------------------------
    def facet_BC(mesh:dolfinx.mesh.Mesh
                 ,facet_tag:dolfinx.mesh.MeshTags
                 ,submesh:dolfinx.mesh.Mesh
                 ,vertex_maps:np.ndarray
                 ,specs:list)->dolfinx.mesh.MeshTags:
        """From the list of facets of the global mesh, generate the mesh tag of the subdomain

        Parameters
        ----------
            mesh (dolfinx.mesh.Mesh): Global mesh object storing the information of the main mesh
            facet_tag (dolfinx.mesh.MeshTags): The facet tags object from the global mesh
            submesh (dolfinx.mesh.Mesh): Submesh object 
            vertex_maps (numpy.ndarray): The maps of the vertex of the mesh tag
            specs (list): a 2D list containing [facet_tag_id_global_mesh,newIDsubmesh]

        Returns
        ----------
            FT: (dolfinx.mesh.MeshTags): the mesh tag object of the subdomain
            
        Source of portion of the code, and general explanation    
        https://fenicsproject.discourse.group/t/how-to-define-bcs-on-boundaries-of-submeshes-of-a-parent-mesh/5470/3

        """
        
        # preparing the lists
        chosen_total = []
        
        val_total = []
        # 
        for boundary, m_id in specs:
            ch_f, val = extract_facet_boundary(mesh, facet_tag, submesh, vertex_maps, boundary, m_id)
            chosen_total.extend(ch_f)
            val_total.extend(val)
        # 
        fac = np.asarray(chosen_total, dtype=np.int32)
        # 
        val = np.asarray(val_total, dtype=np.int32)
        # Create the mesh tag for the given domain
        FT = meshtags(submesh, 1, fac, val)
        
        return FT
    #--------------------------------------------------------------
    # Find the chosen markers 
    marked_cells = []
    
    for marker in phase_set:
        marked_cells.extend(mesh_tag.find(marker))

    marked_cells = np.array(marked_cells, dtype=np.int32)
    # 
    submesh, entity_maps, vertex_maps, node_maps = create_submesh(mesh, mesh.topology.dim, marked_cells)
    
    submesh.topology.create_connectivity(2, 0)
    submesh.topology.create_connectivity(1, 0)
    submesh.topology.create_connectivity(2, 1)
    submesh.topology.create_connectivity(1, 2)
    
    # Facet marker original mesh -> 
    # Select subduction: 
    if name == 'Subduction': 
        # top subduction 8-9 For the subduction subdomain, the tag of the subdcution 
        # are the entire top surface 
        # -- 
        specs = [([8, 9],1), ([6],2), ([7],3), ([5],4)] # [8,9] are the top subduction[6] is the bottom subduction, [7] is the left side of the subduction, [5] is the right side of the subduction
        dict_local = {
            'top_subduction' : 1, # Top subduction
            'bot_subduction' : 2, # Right side of the subduction
            'inflow'         : 3, # Bottom side of the subduction
            'outflow'        : 4, # Left side of the subduction
        }
        # --
        bc = facet_BC(mesh, facet_tag, submesh, vertex_maps, specs)
        # -- 
    elif name == 'Wedge':
        
        specs = [([11],1), ([4],2), ([3],3), ([9],4)] # [1] is the top, [2] is the right side of the wedge, [3] is the bottom side of the wedge, [4] is the left side of the wedge
        dict_local = {
            'overriding'    : 1, # Top subduction
            'right'         : 2, # Right side of the subduction
            'bottom'        : 3, # Bottom side of the subduction
            'slab'          : 4, # Left side of the subduction
        }
        bc = facet_BC(mesh, facet_tag, submesh, vertex_maps, specs)

    elif name == 'Lithosphere':
        
        specs = [([11],1), ([1],2), ([2],3), ([8],4)] # [1] is the top, [2] is the right side of the wedge, [3] is the bottom side of the wedge, [4] is the left side of the wedge
        dict_local = {
            'overriding'    : 1, # Top subduction
            'right'         : 2, # Right side of the subduction
            'bottom'        : 3, # Bottom side of the subduction
            'slab'          : 4, # Left side of the subduction
        }

        bc = facet_BC(mesh, facet_tag, submesh, vertex_maps, specs)

    # Create the functionsubspace for the subdomain
    
    Sol_Spaceph         = fem.functionspace(submesh, ("DG", 0))      # Material ID function space # Defined in the cell space {element wise} apperently there is a black magic that

    
    ph = fem.Function(Sol_Spaceph)
    ph.x.name = "phase"
    
    # Interpolate phase into submesh 
    ph.interpolate(phase, cells0=entity_maps, cells1=np.arange(len(entity_maps)))
    # Create the domain sub-class
    domain = Domain(hierarchy = 'Child', mesh = submesh, cell_par = entity_maps, node_par = vertex_maps, facets = bc , phase = ph,solPh = Sol_Spaceph , bc_dict = dict_local)
    
    return domain
#------------------------------------------------------------------------------------------------------
def read_mesh(ioctrl:IOControls
              ,sc:Scal)->tuple([dolfinx.mesh.Mesh,dolfinx.mesh.MeshTags,dolfinx.mesh.MeshTags]):

    """read the .msh file, and convert into a dolfinx mesh object and extract mesh tags from .msh
    Parameter
    ----------
        ioctrl : IOControls
            Input/Output controls object, stores the information of the path of the .msh file
        sc     : Scal
            Scal object containing the scaling parameters

    Returns
    ----------
        mesh : dolfinx.mesh.Mesh
            dolfinx mesh 
        cell_markers: dolfinx.mesh.MeshTags
            cell markers (i.e., physical surface tags)
        facet_markers : dolfinx.mesh.MeshTags
            facet markers (i.e., physical line tags )

    """


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # 0, 1, ..., size-1

    mesh_name = os.path.join(ioctrl.path_save,'%s.msh'%ioctrl.sname)

    mesh, cell_markers, facet_markers = gmshio.read_from_msh(mesh_name, MPI.COMM_WORLD, gdim=2)
    
    if rank == 0: 
        # Read in mesh
        mesh_name = os.path.join(ioctrl.path_save,'%s.msh'%ioctrl.sname)
        msh = meshio.read(mesh_name)
        
        pt_save = ioctrl.path_save
        if not os.path.exists(pt_save):
            os.makedirs(pt_save)
    
        pt_save = os.path.join(pt_save, ioctrl.sname)
        if not os.path.exists(pt_save):
            os.makedirs(pt_save)    
        
        # Create and save one file for the mesh, and one file for the facets
        triangle_mesh = create_mesh_fenicsx(msh, "triangle", prune_z=True)
        line_mesh = create_mesh_fenicsx(msh, "line", prune_z=True)
        meshio.write("%s/mesh.xdmf"%pt_save, triangle_mesh) # Debug
        meshio.write("%s/mt.xdmf"%pt_save, line_mesh) # Debug 
        # Remove gmsh file, to save memory: every information of the mesh is already known by fenicsx
        os.remove(mesh_name)
        
    mesh = _scaling_mesh(mesh,sc)

    return mesh, cell_markers, facet_markers
#------------------------------------------------------------------------------------------------------
def create_mesh_object(sc:Scal
                       ,ioctrl:IOControls
                       ,g_input:Geom_input)->Mesh:    
    """
    Create a subdomain mesh from the global mesh and interpolate phase information.

    This function extracts a submesh corresponding to a given set of cell markers
    (phases) from the global mesh. The phase function defined on the global mesh is
    then interpolated or transferred onto the subdomain mesh.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The global computational mesh.
    mesh_tag : dolfinx.mesh.meshtags.MeshTags
        Cell tags of the global mesh, typically used to identify material phases.
    facet_tag : dolfinx.mesh.meshtags.MeshTags
        Facet tags of the global mesh, defining boundary features (e.g. slab top surface).
    phase_set : list[int]
        List of cell marker IDs that define the subdomain to extract.
    name : str
        Name of the subdomain (used for identification and debugging).
    phase : dolfinx.fem.Function
        Cell-wise (or DG) function storing phase/material information on the global mesh.

    Returns
    -------
    Domain
        A `Domain` object containing the submesh, associated tags, and interpolated
        phase information.
    """
    
    from stonedfenicsx.scal import dimensionless_ginput
    
    mesh, cell_markers, facet_markers = read_mesh(ioctrl, sc)
    
    Pph           = fem.functionspace(mesh, ("DG", 0))      # Material ID function space # Defined in the cell space {element wise} apperently there is a black magic that 

    # Define the material property field

    phase           = fem.Function(Pph) # Create a function to hold the phase information
    phase.x.name    = "phase"
    phase           = assign_phases(dict_surf, cell_markers, phase) # Assign phases using the cell tags and physical surfaces -> i.e. 10000 = Mantle ? is going to assign unique phase to each node? 

    # Correct the phase: 
    phase.x.array[:] -= 1 # Rather necessary remember to put plus one once you publish it 

    #-- Create additional facet for the shear heating. Since this hell is requiring a lot of useless work, 
    #-- I need to create a yet another ad hoc function for this. 
    # -- 
    domainG = Domain(
            mesh=mesh,
            facets=facet_markers,
            Tagcells=cell_markers,
            phase=phase,
            solPh=Pph,
            bc_dict=dict_tag_lines,
            )    
    # Subducting plate domain
    domainA = create_subdomain(mesh, cell_markers, facet_markers, [1,2]  , 'Subduction',  phase)
    # Wedge plate domain    
    domainB = create_subdomain(mesh, cell_markers, facet_markers, [3]    , 'Wedge',       phase)
    # Overriding plate domain 
    domainC = create_subdomain(mesh, cell_markers, facet_markers, [4,5,6], 'Lithosphere', phase)

    # -- Fill the Mesh object

    MESH = Mesh(g_input = dimensionless_ginput(g_input,sc)
                ,domainG = domainG
                ,domainA = domainA
                ,domainB = domainB
                ,domainC = domainC
                ,comm = MPI.COMM_WORLD
                ,rank = MPI.COMM_WORLD.Get_rank()
                ,element_p = None
                ,element_PT = None
                ,element_V =None)

    return MESH
#-----------------------------------------------------------------------------------------------

    
