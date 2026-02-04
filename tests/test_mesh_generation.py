from stonedfenicsx.numerical_control import NumericalControls,IOControls 
from stonedfenicsx.create_mesh.aux_create_mesh import Geom_input 
from stonedfenicsx.scal import Scal
from stonedfenicsx.utils import Input,print_ph
from stonedfenicsx.create_mesh.create_mesh import create_mesh
from stonedfenicsx.Stoned_fenicx import fill_geometrical_input
import sys
import numpy as np
from numpy.typing import NDArray


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
    
    
    IP = Input()
    # Remove the crustal unit -> plain van keken model
    IP.ocr = 0.0 
    IP.cr = 0.0 
    IP.dl = 10e3
    IP.decoupling = 50e3
    IP.ns_depth = 50e3 
    IP.lit_mt = 50e3 
    IP.lab_d = 50e3
    IP.van_keken = 1 
        
    g_input = fill_geometrical_input(IP) 
    
    
    sc = Scal(L=600e3,Temp=1000,stress=1e9,eta=1e21)

    ctrl = NumericalControls()
    # Enforcing van keken benchmark 
    ctrl.van_keken = IP.van_keken 
    
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
    
    
    assert (la+lb+lc==91427) 
    assert (lg==90694) 
    assert (num_cell ==0) 
    assert len(M.domainG.Tagcells.indices) == 180247

def test_van_keken_mesh_crust():
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
    
    
    IP = Input()
    # Remove the crustal unit -> plain van keken model
    IP.ocr = 6e3 
    IP.cr = 30e3 
    IP.dl = 10e3
    IP.decoupling = 50e3
    IP.ns_depth = 50e3 
    IP.lit_mt = 50e3 
    IP.lab_d = 50e3
    IP.van_keken = 1    
    g_input = fill_geometrical_input(IP) 
    
    
    sc = Scal(L=600e3,Temp=1000,stress=1e9,eta=1e21)

    ctrl = NumericalControls()
    # Enforcing van keken benchmark 
    ctrl.van_keken = IP.van_keken  
    
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
    
    
    assert (la+lb+lc==91699) 
    assert (lg==90964) 
    assert (num_cell ==0) 
    assert len(M.domainG.Tagcells.indices) == 180784


def test_van_keken_mesh_decoupling():
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
    
    
    IP = Input()
    # Remove the crustal unit -> plain van keken model
    IP.ocr = 6e3 
    IP.cr = 30e3 
    IP.dl = 10e3
    IP.decoupling = 80e3
    IP.ns_depth = 50e3 
    IP.lit_mt = 50e3 
    IP.lab_d = 50e3
    IP.van_keken = 1    
    g_input = fill_geometrical_input(IP) 
    
    
    sc = Scal(L=600e3,Temp=1000,stress=1e9,eta=1e21)

    ctrl = NumericalControls()
    # Enforcing van keken benchmark 
    ctrl.van_keken = IP.van_keken
    
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
    
    
    assert (la+lb+lc==91874) 
    assert (lg==91138) 
    assert (num_cell ==0) 
    assert len(M.domainG.Tagcells.indices) == 181132


def test_van_keken_mesh_curved():
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
    
    
    IP = Input()
    # Remove the crustal unit -> plain van keken model
    IP.ocr = 6e3 
    IP.cr = 30e3 
    IP.dl = 10e3
    IP.decoupling = 80e3
    IP.ns_depth = 50e3 
    IP.lit_mt = 50e3 
    IP.lab_d = 50e3
    IP.van_keken = 0
        
    g_input = fill_geometrical_input(IP) 
    
    
    sc = Scal(L=600e3,Temp=1000,stress=1e9,eta=1e21)

    ctrl = NumericalControls()
    # Enforcing van keken benchmark 
    ctrl.van_keken = IP.van_keken 
    
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
    
    
    assert (la+lb+lc==101090) 
    assert (lg==100280) 
    assert (num_cell ==0) 
    assert len(M.domainG.Tagcells.indices) == 199270

if __name__ == '__main__': 
    test_van_keken_mesh()
    test_van_keken_mesh_crust()
    test_van_keken_mesh_decoupling()
    test_van_keken_mesh_curved()