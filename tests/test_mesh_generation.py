from stonedfenicsx.numerical_control import NumericalControls,IOControls 
from stonedfenicsx.aux_create_mesh import Geom_input 
from stonedfenicsx.scal import Scal
from stonedfenicsx.create_mesh import create_mesh
import sys
import numpy as np



def test_van_keken_mesh():
    """_summary_: Test for checking the generation of the van keken geometry for the
    benchmark. 
    -> Call the the geometric input structure -> fill up with the default parameter
    -> Create controls unit 
    -> Create Io-Controls
    -> Create and scale mesh 
    -> Check the number of nodes for the global, and sub meshes 
    -> Check the number of cells for the global and sub meshes: 
       a. check if the number of cells is the same of the given 
       b. check if the number of cells of the submeshes is equal to the global mesh
    NB: number of nodes between submeshes is always not summing up to the global mesh
    """
    g_input = Geom_input(x =np.array([0,660e3]),
                         y=np.array([-600e3,0]),
                         cr = 0.0,
                         ocr = 0.0,
                         lit_mt = 50e3,
                         lc = 0.0, 
                         wc = 2.0e3,
                         slab_tk = 130e3,
                         decoupling=50e3,
                         trans = 10e3,
                         lab_d = 50e3,
                         ns_depth=50e3)
    sc = Scal(L=600e3,Temp=1000,stress=1e9,eta=1e21)

    ctrl = NumericalControls()
    # Enforcing van keken benchmark 
    ctrl.van_keken = 1 
    
    io_ctrl = IOControls(test_name='debug',path_save=sys.path[0], sname='debug_mesh')

    M = create_mesh(io_ctrl, sc,g_input,ctrl)

    XA = M.domainA.mesh.geometry.x 
    XB = M.domainB.mesh.geometry.x 
    XC = M.domainC.mesh.geometry.x
    XG = M.domainG.mesh.geometry.x 
    
    la = len(XA[:,0])
    lb = len(XB[:,0])
    lc = len(XC[:,0])
    lg = len(XG[:,0])
    
    num_cell = len(M.domainG.Tagcells.indices)-(len(M.domainA.cell_par)+len(M.domainB.cell_par)+len(M.domainC.cell_par))
    
    
    assert (la+lb+lc==91497) 
    assert (lg==90764) 
    assert (num_cell ==0) 
    assert len(M.domainG.Tagcells.indices) == 180387
    


    
    
    