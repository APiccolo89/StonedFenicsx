"""
Module that extract the data from the output of StonedFEniCSx
"""

import os
import warnings

import dataclasses
from dataclasses import field,InitVar,dataclass
import h5py
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from global_var import _R_GRID_,_T_CUT_OFF_LIT,_V_CONVERSION_KM_MYR_,_POINT_FILTER_OVERRIDING

import numpy as np
from scipy.interpolate import griddata
from shapely import contains_xy as scontains_xy
from shapely.geometry import Polygon as sPolygon

r_grid = _R_GRID_

# --- 
def get_nested_attr(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj
# --- 
def linear_interpolation(a:float,b:float,c:float,d:float,x:float)->float:
    """Ignorant interpolator
        (x-c) = (d-c)/(b-a) * (v-a) 
        (x-c) * ((d-c)/(b-a))**-1 = v-a
        v = (b-a)/(d-c) * (x-c) + a
        Given the length of the slab: assuming that is a linear segment, 
        compute the coordinate x/y (v) associated with the specific length 
    Args:
        a (float): the unknown value lower bound 
        b (float): the unknown value upper bound
        c (float): the known value lower bound
        d (float): the known value upper bound
        x (float): the known value

    Returns:
        float: the actual value of the interpolation
    """
    
    return (b-a)/(d-c) * (x-c) + a 
# ---
class Data_Raw:
    def __init__(self, f: str, num: int, td: bool = False):
        """_summary_

        Args:
            f (str): path 2 the file
            num (int): number of nodes of the unstructured grid
            td (bool, optional): _description_. Defaults to False. time dependent flag: True: Time Dependent; False: Steady state
        """
        if not td:
            self.SteadyState = Data_experiment(f, num, ts=False)
        else:
            self.TimeDependent = Data_experiment(f, num, ts=True)
    
    def get(self,tag:str):
        if tag not in('TimeDependent','SteadyState'):
            return ValueError(f'There is not attribute {tag!r}')
        return getattr(self,tag) 
# --- 
class Data_experiment:
    """
    Class to extract the data from either the steady state or time dependent h5 file.
    the init function requires the path to the test, the number of points and the number of timestep
    The main issue is for the timedependent case, as I need to extract the number of timesteps from the
    the h5 file, but I will create the function later, and will be in the metadata field of the test.
    times     = time array
    time_list = time list
    Temp      = Temperature array [C]
    Pres      = Dynamic Pressure array [GPa]
    LitPres   = Lithostatic Pressure array [GPa]
    vx        = velocity along x [cm/yr]
    vy        = velocity along y [cm/yr]
    qx        = heat flux x component [W/m]
    qy        = heat flux y component [W/m]
    kappa     = Thermal diffusivity [mm^2/s]
    alpha     = Thermal expansivity
    eta       = Viscosity [Pas]
    rho       = Density [kg/m3]
    k         = Conductivity [W/K/m]
    Cp        = Heat capacity [J/kg/K]
    NoTD      = False
    method:
    extract_data() method that read the output files of StonedFenicsx and fill up all the arrays previously preparred.


    """

    def __init__(self, f: str, num: int, ts: bool):
        """Extract the steady state data from the h5 file.

        The function infers if the test is time dependent, extract the number of timestep and
        allocate arrays with dimension [num,TS], where num is the total number
        of nodes, and TS the total amout of timestep. Then it calls the routine
        to extract the data that automatically generate all the array.

        Args:
            f (str)  : path to the test
            num (int): number of points
            ts (int) : time step
        """
        import h5py
        import numpy as np

        self.times = None
        self.time_list = None
        self.Temp = None
        self.Pres = None
        self.LitPres = None
        self.vx = None
        self.vy = None
        self.qx = None
        self.qy = None
        self.kappa = None
        self.alpha = None
        self.eta = None
        self.rho = None
        self.k = None
        self.Cp = None
        self.NoTD = False

        if ts:
            # Rather necessary as the h5 file was not created in reasonable way
            # Direct to the time dependent file
            if os.path.exists("%s/time_dependent.h5" % f):
                print("The file exists.")
            else:
                print(
                    "The file does not exist. Either the time dependent simulation was not run or the path is incorrect."
                )
                self.NoTD = True
                return None

            fl = h5py.File("%s/time_dependent.h5" % f, "r")
            field = "Function/Temperature  [degC]"
            times = list(fl[field].keys())
            time_list = [float(s.replace("_", ".")) for s in times]
            time_sort = np.argsort(time_list)
            time_list = [time_list[i] for i in time_sort]
            times = [times[i] for i in time_sort]
            TS = len(times)
            self.times = times
            self.time_list = time_list
            fl.close()
        else:
            TS = 0

        self.Temp = np.zeros([num, TS], dtype=float)
        self.Pres = np.zeros([num, TS], dtype=float)
        self.LitPres = np.zeros([num, TS], dtype=float)
        self.vx = np.zeros([num, TS], dtype=float)
        self.vy = np.zeros([num, TS], dtype=float)
        self.qx = np.zeros([num, TS], dtype=float)
        self.qy = np.zeros([num, TS], dtype=float)
        self.kappa = np.zeros([num, TS], dtype=float)
        self.alpha = np.zeros([num, TS], dtype=float)
        self.eta = np.zeros([num, TS], dtype=float)
        self.rho = np.zeros([num, TS], dtype=float)
        self.k = np.zeros([num, TS], dtype=float)
        self.Cp = np.zeros([num, TS], dtype=float)

        self._extract_data(f, ts)

    def _extract_data(self, f: str, ts: bool):

        import h5py
        import numpy as np

        if ts:
            # Direct to the time dependent file
            with h5py.File("%s/time_dependent.h5" % f, "r") as fl:
                for it, time in enumerate(self.times):
                    field_temp = "/Function/Temperature  [degC]/%s" % time
                    field_pres = "/Function/Pressure  [GPa]/%s" % time
                    field_litpres = "/Function/Lit Pres  [GPa]/%s" % time
                    field_v = "/Function/Velocity  [cm/yr]/%s" % time

                    self.Temp[:, it] = np.array(fl[field_temp]).flatten()
                    self.Pres[:, it] = np.array(fl[field_pres]).flatten()
                    self.LitPres[:, it] = np.array(fl[field_litpres]).flatten()

                    v = np.array(fl[field_v])
                    self.vx[:, it] = v[:, 0]
                    self.vy[:, it] = v[:, 1]

        else:
            try:
                f1 = "%s/Steady_state.h5" % f
                fl = h5py.File(f1, "r")
                fl.close()
                proceed = True
            except:
                string = f"Steady state solution does not exist in {f}. Either has not be run or there is a problem with the path"
                warnings.warn(string)
                proceed = False

            if proceed:
                with h5py.File(f1, "r") as fl:
                    self.Temp = np.array(
                        fl["/Function/Temperature  [degC]/0"]
                    ).flatten()

                    self.Pres = np.array(fl["/Function/Pressure  [GPa]/0"]).flatten()

                    self.LitPres = np.array(fl["/Function/Lit Pres  [GPa]/0"]).flatten()

                    v = np.array(fl["Function/Velocity  [cm/yr]/0"])

                    self.vx = v[:, 0]

                    self.vy = v[:, 1]

                    qS = np.array(fl["Function/Heat flux [W/m2]/0"])

                    self.qx = qS[:, 0]

                    self.qy = qS[:, 1]

                    self.Cp = np.array(fl["/Function/Cp  [J/kg]/0"]).flatten()

                    self.k = np.array(fl["/Function/k  [W/m/K]/0"]).flatten()

                    self.rho = np.array(fl["/Function/Density  [kg/m3]/0"]).flatten()

                    self.eta = np.array(fl["/Function/Viscosity  [Pa.s]/0"]).flatten()

                    self.alpha = np.array(fl["/Function/alpha  [1/K]/0"]).flatten()

                    self.kappa = np.array(fl["/Function/kappa  [m2/s]/0"]).flatten()
# --- 
@dataclass(init=True)
class MeshData:
    """ MeshData
    pt: InitVar[str] = ''
    td: InitVar[bool] = False
    
    mesh_tag: the tags of the internal boundaries
    xi: regular grid x-array 
    yi: regulard grid y-array
    Xi: Meshgrid of the regular grid
    Yi: Meshgrid of the regular grid
    ar_s: booelan array containing the points belonging 
          to the subduction domain.
    ar_dom: boolean array containing the domain points bounded 
            by the bottom of the slab
    X: original array containing the points of the triangular
       unstructured grid
    ind_topSlab: indeces of the top of the slab. They must be used
                 with **X** the unstructured grid. 
    ind_Oceanic:NDArray[int] = indeces of the oceanic crust moho. They must be used
                 with **X** the unstructured grid.
    ind_botSlab:NDArray[int] = indeces of the bottom of the slab. They must be used
                 with **X** the unstructured grid.
    """

    pt: InitVar[str] = ''
    td: InitVar[bool] = False
    
    mesh_tag: NDArray[float] = field(init=False)
    xi: NDArray[float] = field(init=False)
    yi: NDArray[float] = field(init=False)
    Xi: NDArray[float] = field(init=False)
    Yi: NDArray[float] = field(init=False)
    ar_s: NDArray[bool] = field(init=False)
    ar_dom: NDArray[bool] = field(init=False)
    X: NDArray[float] = field(init=False)
    ind_topSlab:NDArray[int] = field(init=False)
    ind_Oceanic:NDArray[int] = field(init=False)
    ind_botSlab:NDArray[int] = field(init=False)
    # --- 
    @staticmethod
    def _correct_path(pt:str,td:bool)->Path:
        """From the path to the main test folder, 
        extract the real dataset to read

        Args:
            pt (str) : path to the main folder
            td (bool): flag to tell if the simulation is steady state or not

        Returns:
            pt: the data path ** to check if the memory address of the pt will be the same ** 
        """
        pt =  Path(f"{pt}")
        if td:
            pt = pt / 'time_dependent.h5'
        else: 
            pt = pt / 'Steady_state.h5'
        
        return pt 
    # --- 
    @staticmethod
    def get_coordinate_decoupling(sl_ar:NDArray,dc_y:float)->NDArray:
        """From the decoupling depth, compute the coordinate of the decoupling
        point (x_dc,y_dc)
        This function requires the top surface slab array. 
        Args:
            sl_ar (NDArray): top surface of the slab
            dc_y (float): input decoupling depth

        Returns:
            NDArray: array containing the full coordinate of the decoupling point
        """

                
        # Find the 
        ind = np.flatnonzero(sl_ar[:,1]<dc_y)[0]
        # Find the coordinate of the x
        x0 = sl_ar[ind-1,0]
        x1 = sl_ar[ind,0]
        y0 = sl_ar[ind-1,1]
        y1 = sl_ar[ind,1]
        x = linear_interpolation(x0,x1,y0,y1,dc_y)
        
        return np.array([x,dc_y],dtype=np.float32)
    # --- 
    def collect_coordinate(self,tag:str)->NDArray:
        """Generate a sorted array as a function 
        of the tag

        Args:
            tag (str): string containing the boundary that is required 

        Returns:
            NDArray,NDArray: array with the boundary coordinate, ordered as a function of the x coordinates.
                           : array containing the cumulative length of the array. 
        """

        tags = getattr(self,tag)
        # Extract x coordinate and y coordinate
        coord_x_x = self.X[tags,0]
        coord_x_y = self.X[tags,1]
        # order the coordinate: 
        ind = coord_x_x.argsort()
        coord_x_x = coord_x_x[ind]
        coord_x_y = coord_x_y[ind]
        # create the array 
        coord_x = np.zeros([len(coord_x_x),2],dtype=np.float32)
        coord_x[:,0] = coord_x_x[:]
        coord_x[:,1] = coord_x_y[:]    
        # Compute the length of given array: 
        ell = np.cumsum(np.sqrt(np.diff(coord_x_x)**2
                                +np.diff(coord_x_y)**2))
        # Pre-append 0 to the ell array
        ell = np.concatenate(([0],ell))

        return coord_x, ell

    # ---
    @staticmethod
    def _read_mesh(pt:Path)->tuple[NDArray,NDArray]:
        """Read the mesh and the mesh tag for the given file

        Args:
            pt (Path): Updated path 

        Returns:
            X : array containing the unstructured triangular mesh data
            mesh_tag: array containing the unstructured mesh data of the tags (i.e., numeric value
            associated with specific boundary.)
        """
        with h5py.File(pt, "r") as fl:
            X = np.array(fl["/Mesh/mesh/geometry"])
            
            mesh_tag = np.array(fl["Function/MeshTAG/0"])
        
        return X, mesh_tag.flatten()  
    # ---
    def _create_regular_grid(self):
        """Create the quadrangular grid for visualisation purpose 
        """
        self.xi = np.linspace(np.min(self.X[:, 0]), np.max(self.X[:, 0]), r_grid)

        self.yi = np.linspace(np.min(self.X[:, 1]), np.max(self.X[:, 1]), r_grid)

        self.Xi, self.Yi = np.meshgrid(self.xi, self.yi)        
    # --- 
    def _create_polygon_slab(self):
        """Generate the polygon of the subduction domain,
        this allow to filter out all the other portion of the 
        numerical domain, and just focus on the slab. 
        """

        def extract_coordinates(mesh_tag: list) -> np.ndarray:

            if len(mesh_tag) == 2:
                condition = (self.mesh_tag == mesh_tag[0]) | (
                    self.mesh_tag == mesh_tag[1]
                )
            else:
                condition = self.mesh_tag == mesh_tag[0]

            xbuf = self.X[condition, 0]

            ybuf = self.X[condition, 1]

            sort = np.argsort(xbuf)

            X = np.array([xbuf[sort], ybuf[sort]])

            X = np.transpose(X)

            return X

        # Left inlet:
        X_L = extract_coordinates([7])
        # Bottom slab:
        X_BS = extract_coordinates([6])
        #
        X_BT = extract_coordinates([5])
        # Slab
        X_Sl = extract_coordinates([9, 8])

        a = X_Sl[:, 0]
        b = X_Sl[:, 1]
        a = a[::-1]
        b = b[::-1]
        X_Sl[:, 0] = a
        X_Sl[:, 1] = b


        polygon = sPolygon(
            np.vstack((np.array(X_L), np.array(X_BS), np.array(X_BT), np.array(X_Sl)))
        )

        self.ar_s = scontains_xy(polygon, self.Xi, self.Yi)
    # ---
    def _create_polygon(self):
        """Create the polygon containing the numerical domain
        in case the domain is not **full**. 
        If the flag in the input_test.yaml  **model_full** in geometry
        is true, the model is a quadrangular domain. Otherwise the left boundary 
        is bounded by the inlet and the bottom of the slab. 
        """

        x_min = np.min(self.X[:, 0])

        x_max = np.max(self.X[:, 0])

        y_min = np.min(self.X[:, 1])

        y_max = np.max(self.X[:, 1])

        x = self.X[:, 0]

        y = self.X[:, 1]

        top = np.array([x[self.X[:, 1] == y_max], y[self.X[:, 1] == y_max]])

        bottom = np.array([x[self.X[:, 1] == y_min], y[self.X[:, 1] == y_min]])

        left = np.array([x[self.X[:, 0] == x_min], y[self.X[:, 0] == x_min]])

        right = np.array([x[self.X[:, 0] == x_max], y[self.X[:, 0] == x_max]])

        l_min = np.min(left[1, :])

        p0 = np.array([x_min, l_min])

        p_list = []

        p_list.append(p0)

        xbt = self.X[(self.mesh_tag == 6.0), 0]
        ybt = self.X[(self.mesh_tag == 6.0), 1]
        sort = np.argsort(xbt)

        p_list = np.array([xbt[sort], ybt[sort]])

        p_list = p_list.transpose()

        bottom = bottom.transpose()

        right = right.transpose()

        top = top.transpose()

        left = left.transpose()
        # order bottom boundary

        x_bottom = bottom[:, 0]

        ind_arg = np.argsort(x_bottom)

        bottom = bottom[ind_arg, :]
        # order right boundary

        y_right = right[:, 1]

        ind_arg = np.argsort(y_right)

        right = right[ind_arg, :]
        # order top boundary

        x_top = top[:, 0]

        ind_arg = np.argsort(x_top)

        top = top[ind_arg[::-1], :]

        polygon = sPolygon(
            np.vstack(
                (
                    np.array(p_list),
                    np.array(bottom),
                    np.array(right),
                    np.array(top),
                    np.array(left),
                )
            )
        )

        self.ar_dom = scontains_xy(polygon, self.Xi, self.Yi)
    # --- 
    def __post_init__(self,pt,td):

        pt = self._correct_path(pt,td)
        
        self.X,self.mesh_tag = self._read_mesh(pt)
        
        self._create_regular_grid()
        
        self._create_polygon_slab()
        
        self._create_polygon()

        self.ind_topSlab = (self.mesh_tag == 8.0) | (self.mesh_tag == 9.0)

        self.ind_Oceanic = self.mesh_tag == 10.0
        
        self.ind_botSlab = self.mesh_tag == 6.0
# --- 
class Test:
    """Class containing the data of the current test.

    path_2_test: the current path of the test at hand
    MeshData: class containing the mesh data
    Data_Raw: class containing the field

    methods:
    _interpolate_data(): function that interpolate the raw data (unstructured triangular grid) to a regular quadrangular
    grid. It uses the additional information provided by the boundary to filter artifact

    _interpolate_data_slab(): function that interpolate the raw data of the simulation and select only the slab domain.
    """

    def __init__(self, path_2_test: str, td: bool = False):
        """Initialisation of the class.

        Args:
            path_2_test (str): the path to the test
            td (bool, optional): _description_.Tell the class if the test is steady state or not. False: means steady state, True: means timedependent
        """
        self.path_2_test = path_2_test
        self.MeshData = MeshData(self.path_2_test, td)
        self.Data_raw = Data_Raw(
            self.path_2_test, num=len(self.MeshData.X[:, 0]), td=td
        )

    def interpolate_data(self, Data_field:str,flag_full:bool)->NDArray:
        """Interpolate the data on a regular grid for visualisation
        Args:
            Data_field (str): field to be interpolated
            flag_full (bool): flag that warns that the model is full or 
            finished at the bottom of the slab. 
        Returns:
            Zi (np.array): interpolated data

            Paraview is not able to visualise unstructured data properly,
            so we need to interpolate the data on a regular grid for visualisation purposes.
            The nan mask is used to mask the data outside the domain, 
            and giving you the illusion that I was able to generate a curved mesh in python.
        """


        Xi = self.MeshData.Xi
        Yi = self.MeshData.Yi

        # Extrac the field data
        values = get_nested_attr(self.Data_raw, Data_field)

        Zi = griddata(self.MeshData.X, values, (Xi, Yi), method="linear")

        if not flag_full:
            Zi[not self.MeshData.ar] = np.nan
            

        return Zi
    
    # ---
    def interpolate_data_slab(self, Data_field, ts=0, steady_state=True):
        """Interpolate the data on a regular grid for visualisation

        Paraview is not able to visualise unstructured data properly,
        so we need to interpolate the data on a regular grid for visualisation purposes.
        The nan mask is used to mask the data outside the domain,
        and giving you the illusion that I was able to generate a curved mesh in python.
        Ahah.

        Args:
            Data_field (str): field to be interpolated
            ts (int): timestep index, used only if `steady_state` is False
            steady_state (bool): if True the field has no time dimension

        Returns:
            np.ndarray: interpolated data (Zi)
        """

        Xi = self.MeshData.Xi
        Yi = self.MeshData.Yi

        # Extrac the field data
        values = get_nested_attr(self.Data_raw, Data_field)
        X = self.MeshData.X

        if steady_state:
            Zi = griddata(X, values, (Xi, Yi), method="linear")
        else:
            val = values[:, ts].copy()
            Zi = griddata(X, val, (Xi, Yi), method="linear")

        Zi[not self.MeshData.ar_s] = np.nan

        return Zi
    
    def getdata(self)->Data_experiment:
        """Small utility function to get the experimental data

        Raises:
            ValueError: _description_

        Returns:
            Data_experiment: _description_
        """
        buf = getattr(self,'Data_raw')
        if hasattr(buf,'TimeDependent'):
            return getattr(buf,'TimeDependent')
        elif hasattr(buf,'SteadyState'):
            return getattr(buf,'SteadyState')
        else: 
            raise ValueError('There not any data saved.')

    @staticmethod
    def _compute_coordinate_point(coord_slab:NDArray,ell_slab:NDArray,dist:NDArray,pt0:NDArray)->NDArray:
        # Assuming to start from 0.0 otherwise introducing an option
        x_pt = np.zeros([len(dist),2],dtype=np.float32)
        x_pt[0,:] = pt0
        icx = np.zeros([len(dist)],dtype=np.int32)
        # Find the points
        for i in range(1,len(dist)):
            if dist[i]>np.max(ell_slab):
                x_pt[i:-1,0] = x_pt[i-1,0]
                x_pt[i:-1,1] = x_pt[i-1,1]
                break 
            
            ind = np.flatnonzero(ell_slab >= dist[i])[0]       
            l0 = ell_slab[ind-1]
            l1 = ell_slab[ind]

            icx[i] = ind

            x_pt[i,0] = linear_interpolation(a=coord_slab[ind-1,0],
                                             b=coord_slab[ind,0],
                                             c=l0,
                                             d=l1,
                                             x=dist[i])

            x_pt[i,1] = linear_interpolation(a=coord_slab[ind-1,1],
                                             b=coord_slab[ind,1],
                                             c=l0,
                                             d=l1,
                                             x=dist[i])
        return x_pt,icx 
        # --- 
    def _create_polygon_ph(self
                   ,crd_s:NDArray
                   ,crd_t:NDArray
                   ,x_pt:NDArray
                   ,x_tg:NDArray
                   ,oc_lt:float)->NDArray:
        def find_condition_array(buf,cnd,x):
            buf_a_x = buf[cnd,0]
            buf_a_y = buf[cnd,1]
            buf_a_x = np.append([buf_a_x],x[0])
            buf_a_y = np.append([buf_a_y],x[1])
            x_buf = np.zeros([len(buf_a_x),2],dtype=np.float32)
            x_buf[:,0] = buf_a_x
            x_buf[:,1] = buf_a_y

            return x_buf


        # Pseudo code:
        # 1. collect point points top slab, ocean crust 
        # 2. insert points in the arrays in the appropriate position
        # 3. find them in the new array, generate polygon layout
        # 4. find the nodes of the grid that belongs to the polygon
        #---
        tip = np.zeros([2,2],dtype=np.float32)
        left = np.zeros([2,2],dtype=np.float32)
        buf_a = crd_s.copy()
        buf_b = crd_t.copy()

        pt = [0.0,oc_lt]

        buf_b = np.vstack([pt, buf_b])          # pt = (x, y), becomes row 0


        cnd_a = (crd_s[:,0]<=x_pt[0]) & (crd_s[:,1]>=x_pt[1])
        cnd_b = (buf_b[:,0]<=x_tg[0]) & (buf_b[:,1]>=x_tg[1])

        buf_a = find_condition_array(buf_a,cnd_a,x_pt)
        buf_b = find_condition_array(buf_b,cnd_b,x_tg)

        p_list = buf_a
        tip[0,:] = buf_a[-1,:]
        tip[1,:] = buf_b[-1,:]
        p_list_b = buf_b[::-1]
        left[0,:] = buf_b[0,:]
        left[1,:] = buf_a[0,:]

        polygon = sPolygon(
            np.vstack(
                (
                    np.array(p_list),
                    np.array(tip),
                    np.array(p_list_b),
                    np.array(left),
                )
            )
        )
        ar = scontains_xy(polygon, self.MeshData.Xi, self.MeshData.Yi)
        return ar 
    
    def _track_phase(self,
                x_pt:NDArray
                ,cdr_s:NDArray
                ,cd_target:NDArray
                ,icx:NDArray
                ,i:int
                ,oc_lt:float):

        # Slope of the segment
        s = -1/((cdr_s[icx[i],1]-cdr_s[icx[i]-1,1])
               /(cdr_s[icx[i],0]-cdr_s[icx[i]-1,0]))
        # Redundant I know
        dy = np.abs(oc_lt) /( -(1+1/s**2)**(1/2))
        dx = dy*(1/s)
        x = dx+x_pt[i,0]
        y = dy+x_pt[i,1]
        # Create Polygon 
        ar = self._create_polygon_ph(cdr_s,cd_target,x_pt[i,:]
                               ,np.array([x,y])
                               ,oc_lt=oc_lt)
        return ar
    # ---

    def _filter_overriding_plate(self,dc_ar:NDArray)->NDArray[bool]:
        """_summary_

        Args:
            dc_ar (NDArray): decoupling coordinate

        Returns:
            NDArray[bool]: the condition that needs to be checked for filtering the lithosphere:
            True: is overriding domain; False: is not overrding domain
        """
        # Find the maximum x value
        x_max = self.MeshData.X[:,0].max()
        # find the slope 
        dy_dx = (_POINT_FILTER_OVERRIDING-dc_ar[1])/(x_max-dc_ar[0])
        
        return self.MeshData.Yi> dc_ar[1] + dy_dx * (self.MeshData.Xi-dc_ar[0])
    
    
    # --- 
    def _create_phase_field(self,
                            x_pt:NDArray,
                           icx:NDArray
                           ,cdr_s:NDArray
                           ,cd_oc:NDArray
                           ,oc_lt:float
                           ,dc:NDArray)->NDArray:
        # Pseudocode
        # To do -> explicit parsing alos the information concerned the crust
        # 1 -> find the perpendicular slope of the segment of the slab 
        # 2 -> find the coordinate of the crust 
        # 3 -> find coordinate bottom of the slab
        #    2&3: compute slope
        #       : compute the inverse of the slope
        #       : transform coordinate 
        #       : set up system of equation and find out that: 
        #       : dy = - dx * 1/s Dio porco, me e la mia memoria -1/s
        #       : d = (dx**2+dy**2)**(1/2)
        #       : dx = dy * s 
        #       : d = (dy*(1+s**2)**(1/2))
        #       : y = d/-(1+s**2)**(1/2)
        #       : x = y*s 
        #       : retransform coordinate 
        #       : create polygon and do the binding 
        # ----- 
        # 4 -> clip ocean/bottom slab 
        # 5 -> create poligon 
        # 6 -> build the slab phase field 
        d = np.array([oc_lt],dtype=np.float32)
        #function inpolygon

        temp = self.interpolate_data('TimeDependent.Temp',True)
        Ph = temp.copy() * 0
        ar = Ph.copy()
        ar = ar[:,:,0]
        ov_ar = self._filter_overriding_plate(dc)
        for i in range(x_pt.shape[0]):
            temp0 = temp[:,:,i]
            if i>0 :
                ar_s = self._track_phase(x_pt=x_pt
                                    ,cdr_s=cdr_s
                                    ,cd_target=cd_oc
                                    ,icx=icx
                                    ,i=i
                                    ,oc_lt=oc_lt)
                ar[(temp0<_T_CUT_OFF_LIT)] = 1
                ar[(ar==1) & ( self.MeshData.ar_s==0) & (ov_ar==1)] = 0
                ar[ar_s ] = 2
            Ph[:,:,i] = ar[:,:]
        return Ph
    # ---
    def get_phase_field(self,vc:float,oc_tk:float,dc:float)->None:
        """Function that generates a time-dependent 
        phase field, using the information of the mesh
        the temperature evolution. 
        The current function assumes that the only information
        to track is the oceanic crust, while the lithospheric mantle
        is defined using solely the temperature. The temperature cut-off
        is stored in the global_var.
        
        The function works as follow: 
        1) collects the coordinate of the top surface of the slab and the one of 
        oceanic moho. 
        2) Track the point of the subducted crust -> starting point (0,0) 
        3) Compute the cumulative distance travelled by the initial point along the top surface of the slab
        4) Construct the polygon that define the oceanic crust
        5) Filter the data of the model as a function of the temperature, and a **rudimental** geometrical 
        filter that focus all the attention on the slab and extract the points that can be interpreted as litho-
        spheric mantle of the subductin mantle. 
        ---> Filter: use the decoupling point coordinates (x_dc,y_dc), Use the a point on the right boundary (x_f,-300.0)
            -> the data above y>y_dc + m (x-x_dc) are not considered in the temperature based phase transformation 
        
        Args:
            vc (float): velocity of the slab. 
            oc_tk(float): thickness of the oceanic crust. 
            dc(NDArray): y-decoupling point coordinates
        """
        # The function assumes that the dc is always positive, just for making the life easier
        dc = np.abs(dc)
        # Check wheter or not the test is time-dependent or not
        drw = self.Data_raw
        if not hasattr(drw,'TimeDependent'):
            raise ValueError('The phase field can be created only with',
                             'time dependent solutions.')
        else: 
            slab_x, slab_ell = self.MeshData.collect_coordinate(tag= 'ind_topSlab')
            ocean_x, ocean_ell = self.MeshData.collect_coordinate(tag= 'ind_Oceanic')
            # create distance of the top of the surface of the slab
            dist = vc * np.array(drw.TimeDependent.time_list) *_V_CONVERSION_KM_MYR_
            # Compute the coordinate of the top of the slab with time
            x_pt,icx = self._compute_coordinate_point(coord_slab=slab_x
                                    ,ell_slab=slab_ell
                                    ,dist=dist
                                    ,pt0=np.array([0,0]))
            Ph = self._create_phase_field(x_pt=x_pt
                        ,icx=icx
                       ,cdr_s=slab_x
                       ,cd_oc=ocean_x
                       ,oc_lt =oc_tk
                       ,dc=test.MeshData.get_coordinate_decoupling(slab_x,-dc))
            
            return Ph 
            
        
    
def test_mesh():
    """Test to see if the object mesh is created. 
    """
    pt = Path(__file__).resolve().parents[1] 
    
    pt = pt/'test_folder'/'TDexp'/'TDexp_case_debug'
    passed = True
    try:
        mesh = MeshData(pt=str(pt),td=True)
    except: 
        passed = False 
        
    assert passed
    
    


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    pt = Path(__file__).resolve().parents[1] 
    
    pt = pt/'test_folder'/'TDexp'/'TDexp_case_debug'
    
    pt_save = Path(__file__).resolve().parents[1] / 'Output'
    
    pt_save.mkdir(parents=True,exist_ok=True)
    
    test = Test(pt,td=True)
    
    Ph = test.get_phase_field(vc=5.0,oc_tk=6.0,dc=-80.0) 
    Temp = test.interpolate_data('TimeDependent.Temp',True)
    a = []
    b = []
    c = []
    d = []
    e = []
    fig1 = plt.Figure([10,10])
    ax = fig1.gca()
    ts = test.Data_raw.TimeDependent.time_list
    for i,t in enumerate(ts):
        if t > 0: 
            c.remove()
            e.remove()
        
        ax.set_title(f'Time = {t:.2f} Myr')
        c = ax.pcolormesh(test.MeshData.Xi,test.MeshData.Yi,Ph[:,:,i],cmap='inferno')
        e = ax.contour(test.MeshData.Xi,test.MeshData.Yi,Temp[:,:,i],
                       levels=[0,200,400,600,800,1000,1200,1300],
                       colors='w', linewidths=1.0)

        fig1.savefig(f'{pt_save}/random{i}.png')
    
    