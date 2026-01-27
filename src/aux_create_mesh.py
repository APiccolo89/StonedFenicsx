from .package_import import *
from .scal import Scal 

# --- Index 
'''
DICTIONARIES:
dict_surf
dict_tag_lines
CLASS: 
Mesh [Important]
Domain[Important]
Geom_input[Important]
Class_Points[Secondary] => Used in create_mesh
Class_Line[Secondary]   => Used in create_mesh
FUNCTION: 

'''



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

class Mesh(): 
    def __init__(self) :

        ''' g_input: Geometric input parameters,
            domainG: Global domain
            domainA: Subduction zone domain
            domainB: Wedge domain
            domainC: Overriding plate domain
            rank: MPI rank
            size: MPI size
            element_p: Finite element for pressure
            element_PT: Finite element for temperature
            element_V: Finite element for velocity
        '''
        self.g_input         : geom_input                            # Geometric input
        self.domainG         : Domain                                # 
        self.domainA         : Domain
        self.domainB         : Domain
        self.domainC         : Domain
        self.comm            : mpi4py.MPI.Intracomm
        self.rank            : int
        self.size            : int 
        self.element_p       : ufl.FiniteElement   
        self.element_PT      : ufl.FiniteElement
        self.element_V       : ufl.FiniteElement
 
 
@dataclass
class Domain:
    """
    Domain stores the mesh and associated data:
      - cell_par, node_par: parent relationships (if global mesh has submeshes)
      - facets: tagged facets (boundary features)
      - Tagcells: tagged cells (markers)
      - bc_dict: dictionary of boundary condition tags/names
      - solPh: function space for material properties
      - phase: material phase function
      # Admit: corrected with chatgpt, yeah, I am lazy
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
        



#-------------------------------------------------------------------------------------------------
# Class containing the geometric input of the numerical simulation. The values are in meters, and is made dimensionless 
# later on.
#-------------------------------------------------------------------------------------------------

class Geom_input():
    def __init__(
        self,
        x: NDArray[np.float64] = np.array([0.0, 1000e3]),
        y: NDArray[np.float64] = np.array([-660e3, 0.0]),
        cr: float = 20e3,
        ocr: float = 6e3,
        lit_mt: float = 30e3,
        lc: float = 0.5,
        wc: float = 2.0e3,
        slab_tk: float = 130e3,
        decoupling: float = 100e3,
        trans: float = 10e3,
        lt_d : float = 50e3,
        lab_d: float = 50e3) -> None:
         
        self.x                 = x               # main grid coordinate
        self.y                 = y   
        self.slab_tk           = slab_tk
        self.cr                = cr              # crust 
        self.ocr               = ocr             # oceanic crust
        self.lit_mt            = lit_mt          # lithosperic mantle  
        self.lc                = lc              # lower crust ratio 
        self.wc                = wc              # weak zone 
        self.lt_d              = lt_d     # total lithosphere thickness
        self.decoupling        = decoupling      # decoupling depth -> i.e. where the weak zone is prolonged 
        self.resolution_normal = wc  # To Do
        self.theta_out_slab    = []
        self.theta_in_slab     = []
        self.trans             = trans
        if lab_d == 0.0:
            self.lab_d         = self.lt_d
        else:
            self.lab_d         = lab_d
    
    def dimensionless_ginput(self,sc:Scal):
        self.x                 /= sc.L               # main grid coordinate
        self.y                 /= sc.L   
        self.cr                /= sc.L              # crust 
        self.ocr               /= sc.L             # oceanic crust
        self.lit_mt            /= sc.L          # lithosperic mantle  
        self.wc                /= sc.L             # weak zone 
        self.lt_d              /= sc.L    # total lithosphere thickness
        self.decoupling        /= sc.L      # decoupling depth -> i.e. where the weak zone is prolonged 
        self.resolution_normal /= sc.L  # To Do
        self.trans             /= sc.L
        self.lab_d             /= sc.L
        
        return self 
        
#-----------------------------------------------------------------------------------------------------------------
class Class_Points():
    def update_points      (self,
                            mesh_model:gmsh.model,
                            sx        :float,
                            sy        :float,
                            bx        :float,
                            by        :float,
                            oc_cx     :float | None,
                            oc_cy     :float | None,
                            g_input   :Geom_input) -> gmsh.model:
        """
        To Do Friday: document this function



        """

        # Function CREATE POINTS

        #-- Create the subduction,channel and oceanic crust points 
        # Point of subduction -> In general, the good wisdom would tell you to not put a disordered list of points, 
        # but, I am paranoid, therefore: just create the geometry of slab, channel and oceanic crust such that y = f(x) where f(x) is always growing (or decreasing)
        # if you want to be creative, you have to modify the function of create points, up to you, no one is against it. 

        self.max_tag_s,  self.tag_subduction, self.coord_sub,     mesh_model     = _create_points(mesh_model, sx,    sy,    g_input.wc, 0)
        self.max_tag_bots,  self.tag_bottom,    self.coord_bottom, mesh_model     = _create_points(mesh_model, bx,    by,    g_input.wc, self.max_tag_s)
        if g_input.ocr != 0.0: 
            self.max_tag_oc, self.tag_oc,         self.coord_ocean,   mesh_model     = _create_points(mesh_model, oc_cx, oc_cy, g_input.wc, self.max_tag_bots)
        else: 
            self.max_tag_oc = self.max_tag_bots; self.coord_ocean  = None
        # -- Here are the points at the boundary of the model. The size of the model is defined earlier, and subduction zone is modified as such to comply the main geometrical input, 
        # I used subduction points because they define a few important point. 


        self.max_tag_b, self.tag_right_c_b, self.coord_bc,  mesh_model            = _create_points(mesh_model,  g_input.x[1], g_input.y[0],         g_input.wc,  self.max_tag_oc,  True)
        self.max_tag_c, self.tag_right_c_l, self.coord_lr,  mesh_model            = _create_points(mesh_model,   g_input.x[1], -g_input.lt_d, g_input.wc,  self.max_tag_b,  True)
        self.max_tag_d, self.tag_right_c_t, self.coord_top, mesh_model            = _create_points(mesh_model,   g_input.x[1],  g_input.y[1],         g_input.wc,  self.max_tag_c,  True)

        if g_input.cr != 0.0:
            self.max_tag_e, self.tag_right_c_cr, self.coord_crust,    mesh_model  = _create_points(mesh_model,  g_input.x[1], -g_input.cr,                g_input.wc, self.max_tag_d, True)
            if g_input.lc !=0: 
                    self.max_tag_f, self.tag_right_c_lcr,  self.coord_lcr, mesh_model = _create_points(mesh_model,  g_input.x[1], -g_input.cr*(1-g_input.lc), g_input.wc, self.max_tag_e, True)
            else: 
                self.coord_lcr = None
        else: 
            self.coord_lcr = None ; self.coord_crust = None

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
        
        # Top Boundary      
        p_list                                                    = [CP.tag_subduction[0], CP.tag_right_c_t[0]]
        self.max_tag_top, self.tag_L_T, self.lines_T, mesh_model  = _create_lines(mesh_model,0,p_list,False)

        #[right boundary]
        # -- This a tedius job, but remember, dear, to just follow the logic: since I introduced a few new functionality, and the gmsh function are pretty annoying 
        # I tried to make the function as simple as my poor demented mind can do. So, it is not the most easiest possible, but the easiest that I was able to conceive
        # in 10 minutes. Good luck 

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
        i_s = find_tag_line(CP.coord_sub,     g_input.lt_d,'y')
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
        # Plot debug 
        DEBUG = 0 
        if DEBUG == 1: 
            
            for i in range(len(self.line_global[0,:])):
                p0 = self.line_global[0,i]
                p1 = self.line_global[1,i]
                coord_x = [CP.global_points[0,p0-1],CP.global_points[0,p1-1]] 
                coord_y = [CP.global_points[1,p0-1],CP.global_points[1,p1-1]]
                plt.plot(coord_x, coord_y, c='r') 
        
        
        
        
        
        return mesh_model
                
#-----------------------------------------------------------------------------------------------------------------
def create_loop(l_list:list,
                mesh_model:gmsh.model,
                tag:int) -> gmsh.model:
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


def function_create_slab_channel(data_real:bool,
                                 c_phase:Geom_input,
                                 SP=[],
                                 fname=[])-> tuple[NDArray[np.float64],NDArray[np.float64],NDArray[np.float64],NDArray[np.float64],NDArray[np.float64] | None,NDArray[np.float64] | None]: 

    """
    Input: 
        data_real = flag to warn that we are dealing with real data
        wc        = the thickness of the subduction channel [m]
        lt        = lithospheric thickness                  [m]
        SP        = [] default -> Slab class in case
        fname     = [] default -> file name of the real slab 
    Output: 
        ax         = slab x coordinate    [m]
        ay         = slab y coordinate    [m]
        theta_mean = mean theta angle along slab lenght
        cx         = channel x coordinate [m]
        cy         = channel y coordinate [m]
        ex         = extra node x  coordinate at the base of over-
                     riding plate and within channel [m]
        ey         = extra node x  coordinate at the base of over-
                     riding plate and within channel [m]
    
    Explanation: 
        the function create arrays of point that are used to compute the domain. First
        compute the surface of the slab using either the Slab class or the real data. 
        Then compute the subduction channel (alg: per each point of the slab, compute
        the perpendicular projection whose distance is the wc). 
        Then call functions that correct the slab, adding a point at the intersection with 
        the depth of the lithosphere. 
        Then call a function that correct the channel, finding its point at surface and the base 
        of the lithosphere. On top of that, compute the coordinate of an extra node at the 
        base of the lithosphere, between slab and channel. 

    """
    
    # Prepare the slab surface. Slab surface can be derived from real data, or just created ad hoc by the slab routines
    # Add the extra node that is required for the interface with the lithosphere. 
    
    if data_real == True:
        a = np.loadtxt(fname)
        a *= 1e3
        ax = a[:,0]
        ay = -a[:,1]
        # Place holder for a closure to compute the theta associated with a real geometry 
    else:
        SP._find_slab_surface()
        ax = SP.slab_top[:,0]*1e3
        ay = SP.slab_top[:,1]*1e3
        theta_mean = SP.theta_mean 
        
    # Create the channel using the subduction interface as guide
    #cx,cy = function_create_subduction_channel(ax,ay,theta_mean,c_phase)
    if c_phase.ocr != 0.0:
        ox,oy = function_create_oceanic_crust(ax,ay,theta_mean,c_phase.ocr)
    else: 
        ox = None; oy = None 
    # Correct the slab surface and find the extra node
    ax,ay,theta_mean = find_extra_node(ax,ay,theta_mean,c_phase)
    # Correct the subduction channel as well finding the extranode 
    # I needed to correct first the slab and then the channel, because otherwise it was
    # creating an unrealistic bending of the weak zone. Initially I was recomputing the angle 
    # between the linear segment, using an average. It is more convinient use two // lines and then 
    # correcting them. 
    #cx,cy,ex,ey = correct_channel(cx,cy,ax,ay,c_phase)
    bx,by = function_create_subduction_bottom(ax,ay,theta_mean,c_phase.slab_tk)

    return ax,ay,bx,by,ox,oy

#-----------------------------------------------------------------------------------------------------------------

def function_create_subduction_channel(sx:NDArray,
                                       sy:NDArray,
                                       th:float,
                                       c_phase:Geom_input)->tuple[NDArray[np.float64],NDArray[np.float64]]:
    wc = c_phase.wc
    cx = np.zeros([np.amax(sx.shape),1])
    cy = np.zeros([np.amax(sx.shape),1])
    # Loop over the interface of the slab and find the points on the top of the surface of the subduction channel: the point on the of the top of the channel are perpendicular to the slab interface#
    # Compute the top surface of the subduction channel
    cx = sx + wc * np.sin(th)
    cy = sy + wc * np.cos(th)
    # Find the node that are lower than the top boundary
    ind = np.where(cy < sy[0])
    ind = ind[0]
    # find the first node
    ind_min = ind[0]
    # Compute the new x-coordinate of the node that are lower than the top boundary
    x_top_channel = -cy[ind_min-1]/((cy[ind_min] - cy[ind_min-1])/(cx[ind_min] - cx[ind_min-1]))+cx[ind_min-1]
    cxn = np.zeros([len(ind)+1])
    cyn = np.zeros([len(ind)+1])
    cxn[0] = x_top_channel
    cyn[0] = sy[0]
    cxn[1:] = cx[cy < sy[0]]
    cyn[1:] = cy[cy < sy[0]] 
    cx = cxn
    cy = cyn

    # fix x-coordinate of point at y = y_max 
    cy[-1] = np.amin(sy)

    shift_x2 = ((sx[-1] - sx[-2]) * (cy[-1] - cy[-2])) / (sy[-1] - sy[-2]) 
    cx[-1] = cx[-2] + shift_x2

    return cx,cy


def function_create_oceanic_crust(sx:NDArray[np.float64],
                                  sy:NDArray[np.float64],
                                  th:float,
                                  olt:float):

    cx = np.zeros([np.amax(sx.shape),1])
    cy = np.zeros([np.amax(sx.shape),1])
    # Loop over the interface of the slab and find the points on the top of the surface of the subduction channel: the point on the of the top of the channel are perpendicular to the slab interface#
    # Compute the top surface of the subduction channel
    cx = sx - olt*np.sin(th)
    cy = sy - olt*np.cos(th)
    # Find the node that are lower than the left boundary [same function, but switching the position -> ca rotate ]
    cord_x = 0.0 
    cord_z = -olt/np.cos(th[0])
    cy = cy[cx>0.0]
    cx = cx[cx>0.0]
    cx = np.insert(cx,0,0.0)  
    cy = np.insert(cy,0,cord_z)  


    e_node2,t_ex1 = _find_e_node(cx,cy,cx*0.0,-np.min(sy),flag=False)
    cx,cy,t  = _correct_(cx,cy,e_node2,-np.min(sy),cy*0.0,0.0)

    cx_n = cx[(cx>=0.0) & (cy>=np.min(sy))]
    cy_n = cy[(cx>=0.0) & (cy>=np.min(sy))]

    return cx_n,cy_n

#-----------------------------------------------------------------------------------------------------------------


def function_create_subduction_bottom(sx:NDArray[np.float64],
                                      sy:NDArray[np.float64],
                                      th:float,
                                      lt:float)->tuple[NDArray[np.float64],NDArray[np.float64]]:

    cx = np.zeros([np.amax(sx.shape),1])
    cy = np.zeros([np.amax(sx.shape),1])
    # Loop over the interface of the slab and find the points on the top of the surface of the subduction channel: the point on the of the top of the channel are perpendicular to the slab interface#
    # Compute the top surface of the subduction channel
    cx = sx - lt*np.sin(th)
    cy = sy - lt*np.cos(th)
    # Find the node that are lower than the left boundary [same function, but switching the position -> ca rotate ]
    cord_x = 0.0 
    cord_z = -lt/np.cos(th[0])
    cy = cy[cx>0.0]
    cx = cx[cx>0.0]
    cx = np.insert(cx,0,0.0)  
    cy = np.insert(cy,0,cord_z)  


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
                    c_phase:Geom_input)->tuple[NDArray[np.float64],NDArray[np.float64],NDArray[np.float64]]:

    #-- Find nodes 
    e_node_uc,t_ex1 = _find_e_node(ax,ay,t,c_phase.cr*(1-c_phase.lc))
    ax,ay,t = _correct_(ax,ay,e_node_uc,c_phase.cr*(1-c_phase.lc),t,t_ex1)

    
    e_node_lc,t_ex2 = _find_e_node(ax,ay,t,c_phase.cr)
    ax,ay,t = _correct_(ax,ay,e_node_lc,c_phase.cr,t,t_ex2)

    
    e_node_lit,t_ex3 = _find_e_node(ax,ay,t,c_phase.lt_d)
    ax,ay,t = _correct_(ax,ay,e_node_lit,c_phase.lt_d,t,t_ex3)

        
    e_node_lit,t_ex3 = _find_e_node(ax,ay,t,c_phase.decoupling)
    ax,ay,t = _correct_(ax,ay,e_node_lit,c_phase.decoupling,t,t_ex3)


    return ax,ay,t


def correct_channel(cx:NDArray[np.float64],
                    cy:NDArray[np.float64],
                    sx:NDArray[np.float64],
                    sy:NDArray[np.float64],
                    c_phase:Geom_input)->tuple[NDArray[np.float64],NDArray[np.float64],float,float]:
    nr_channel_points = np.amax(cx.shape)
    #-- Find nodes 
    ' Dovevo alternare le cazzo di funzioni, ho perso 5-7 ore del mio tempo, per fortuna non ho il cancro, altrimenti che gioia' 
    
    
    e_node_uc,t_ex1 = _find_e_node(cx,cy,np.zeros_like(cx),c_phase.cr*(1-c_phase.lc))
    cx,cy,t = _correct_(cx,cy,e_node_uc,c_phase.cr*(1-c_phase.lc),cx*0.0,0)

    e_node_lc,t_ex2 = _find_e_node(cx,cy,np.zeros_like(cx),c_phase.cr)
    cx,cy,t = _correct_(cx,cy,e_node_lc,c_phase.cr,cx*0.0,0)

    e_node_lit,t_ex3 = _find_e_node(cx,cy,np.zeros_like(cx),c_phase.lt_d)
    cx,cy,t = _correct_(cx,cy,e_node_lit,c_phase.lt_d,cx*0.0,0)

    e_node_dc,t_ex4 = _find_e_node(cx,cy,np.zeros_like(cx),c_phase.decoupling)
    cx,cy,t = _correct_(cx,cy,e_node_dc,c_phase.decoupling,cx*0.0,0)


   
    cx = cx[cy>=-c_phase.decoupling]
    cy = cy[cy>=-c_phase.decoupling]
    

    # we want to add an extra node in the middle of the channel so that we can easily assign boundary conditions and we control the mesh there
    for i in range (0,len(sx)-1):
        if sy[i] == - c_phase.lt_d:
            slab_x_extra_node = sx[i]
    ex = (cx[e_node_lit] + slab_x_extra_node) / 2
    ey = - c_phase.lt_d


    return cx,cy,ex,ey

#----------------------------------------------------------------------------------------------------------------------------------

def _create_points(mesh,                                            # I am not able to classify 
                   x:float,                                         # coordinate point/points
                   y:float,                                         # coordinate point/points
                   res:float,                                       # resolution of the ppint
                   tag_pr:int,                                      # maximum tag of the previous group
                   point_flag:bool=False)-> tuple[int, int, float]: # a flag to highlight that there is only one point, yes, lame. 
    
    """_summary_: Summary: function that creates the point using gmsh. 
    gmsh   : the gmsh object(?)
    x,y    : coordinate [float]-> can be an array as well as a single point
    res    : the resolution of the triangular mesh 
    tag_pr : previous tag -> i.e., since gmsh is internally assign a tag, I need to keep track of these points for then updating coherently
    Returns:
        max_tag  : -> the maximum tag of function call
        tag_list : -> the list of tag for the point of this function call
        coord    : -> the coordinates -> rather necessary for some functionality later on. Small modification {should I change the name?} -> Add also the tag,  
    
    Problem that solves: for a given list of points (from 1 to N) -> call gmsh function, create points, store information for the setup generation
    -> Nothing sideral, but I try to document everything for the future generations. 
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
    
    len_p = len(tag_p)-1
    tag_l = []
    previous = previous+1
    lines = np.zeros([3,len_p],dtype=np.int32)
    for i in range(len_p): 
        a = tag_p[i]
        b = tag_p[i+1]
        'it appears that the gmsh is an horrible person: i try to give the golden rule that the tag of the points are following an order,'
        'which entails that the minimum of the two point defining a segment is the first -> such that it is always possible determine the fucking order'
 
        tag_1 = np.min([a,b])
        tag_2 = np.max([a,b]) 
        mesh_model.geo.addLine(tag_1,tag_2,previous+i)
        tag_l.append(previous+i)
        lines[0,i] = tag_1 
        lines[1,i] = tag_2 
        lines[2,i] = previous+i
        
    max_tag = np.max(tag_l)
    
    return max_tag,tag_l,lines,mesh_model

# End File ----
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
