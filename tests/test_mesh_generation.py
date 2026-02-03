from stonedfenicsx.numerical_control import NumericalControls,IOControls 
from stonedfenicsx.create_mesh.aux_create_mesh import Geom_input 
from stonedfenicsx.scal import Scal
from stonedfenicsx.utils import Input,print_ph
from stonedfenicsx.create_mesh.create_mesh import create_mesh
import sys
import numpy as np
from numpy.typing import NDArray

def fill_geometrical_input(IP)->Geom_input: 
    from dataclasses import fields
    """_summary_


    Returns:
        _type_: geometrical input dataclasses
         
    """
    
    
    
    g_input = Geom_input
    
    g_input.x = IP.x 
    g_input.y = IP.y
    g_input.cr = IP.cr
    g_input.ocr = IP.ocr
    g_input.lc = IP.lc 
    g_input.lit_mt = IP.lit_mt
    g_input.lab_d = IP.lab_d
    g_input.slab_tk = IP.slab_tk
    g_input.resolution_normal = IP.wc
    g_input.resolution_refine = IP.wc
    g_input.ns_depth = IP.ns_depth
    g_input.decoupling = IP.decoupling
    g_input.sub_constant_flag = IP.van_keken
    g_input.sub_type = IP.slab_type
    g_input.sub_trench = IP.trench 
    g_input.sub_dl = IP.dl 
    g_input.sub_theta_0 = IP.theta0
    g_input.sub_theta_max = IP.theta_max
    g_input.sub_Lb = IP.Lb
    g_input.trans = IP.transition
    g_input.sub_path = IP.sub_path
    g_input.wz_tk = IP.wz_tk
    
    fields_g_input = fields(g_input)
    
    print_ph('----:Geometric input of the numerical simulation: ')
    it = 0 
    for f in fields_g_input: 
        if (f.name != 'theta_out_slab') and (f.name != 'theta_in_slab'):
            values = eval(f'g_input.{f.name:s}')
            if not isinstance(values, np.ndarray):
                if isinstance(values,str):
                    string_2_print = f'          {it}. {f.name:s} = {values:s}'
                else:
                    string_2_print = f'          {it}. {f.name:s} = {values:.3f} [m]'
                    if f.name == 'sub_theta_0' or f.name == 'sub_theta_max' or f.name == 'sub_constant_flag': 
                        string_2_print = f'          {it}. {f.name:s} = {values:.3f} [n.d.]'
            else: 
                string_2_print = f'          {it}. {f.name:s} = [{values[0]:.3f}, {values[1]:.3f}] [m]'
            
            print_ph(string_2_print)            
            
            if f.name == 'sub_theta_0' or f.name == 'sub_theta_max':
                print_ph('                          => converted in rad. ')
                if f.name == 'sub_theta_0':
                    g_input.sub_theta_0 *= np.pi/180.0 
                else: 
                    g_input.sub_theta_max *= np.pi/180.0
            

            if f.name == 'y':
                if g_input.y[0] >= 0.0:
                    raise ValueError(f'minimum coordinate of y = {g_input.y[0]:.3f} is positive or equal to 0.0. The coordinate must be negative')
            if ((f.name == 'ocr') or (f.name == 'cr') or (f.name=='lc') or (f.name =='decoupling') or (f.name =='lit_mit')
                or (f.name =='lab_d') or (f.name=='trans') or (f.name =='resolution_normal') or (f.name =='resolution_refine') 
                or (f.name =='decoupling') or (f.name =='slab_tk')):
                
                if values < 0: 
                    raise ValueError(f'{f.name :s} is negative. The coordinate must be positive')
        it = it+1
        
    print_ph('----://////:---- ')

    return g_input 

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
    
    g_input = fill_geometrical_input(IP) 
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




if __name__ == '__main__': 
    test_van_keken_mesh()