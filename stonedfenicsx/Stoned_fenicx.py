

# -- Main solver script --
from .package_import import *

#---------------------------------------------------------------------------------------------------------
# My modules 
#---------------------------------------------------------------------------------------------------------
from stonedfenicsx.create_mesh.create_mesh     import create_mesh as cm 
from stonedfenicsx.phase_db                    import PhaseDataBase 
from stonedfenicsx.numerical_control           import ctrl_LHS 
from stonedfenicsx.utils                       import print_ph
from stonedfenicsx.phase_db                    import _generate_phase
from stonedfenicsx.scal                        import Scal
from stonedfenicsx.scal                        import _scaling_material_properties
from stonedfenicsx.numerical_control           import NumericalControls
from stonedfenicsx.numerical_control           import IOControls
from stonedfenicsx.create_mesh.create_mesh     import Geom_input
from stonedfenicsx.scal                        import _scaling_control_parameters
from stonedfenicsx.scal                        import _scale_parameters
from stonedfenicsx.solution                    import solution_routine

dict_options = {'NoShear':0,
                'Linear':1,
                'SelfConsistent':2}

def generate_phase_database(IP,Phin)->PhaseDataBase:
    
    pdb = PhaseDataBase(7,IP.phi*np.pi/180)
    
    it = 1 
    for i in dir(Phin):
        if 'Phase' in i:
            phase = getattr(Phin,i)
            print_ph(f"Generating phase {it} : {i}, Phase Name : {phase.name_phase}")
            
            print_ph('-----Rheological Parameters------')
            print_ph(f"Diffusion law  : {phase.name_diffusion if hasattr(phase, 'name_diffusion') else 'Constant'}")
            if phase.Edif != -1e23:
                print_ph(f"   Edif : {phase.Edif} ")
            if phase.Vdif != -1e23:
                print_ph(f"   Vdif : {phase.Vdif} ")
            if phase.Bdif != -1e23:
                print_ph(f"   Bdif : {phase.Bdif} ")
            
            
            print_ph(f"Dislocation law: {phase.name_dislocation if hasattr(phase, 'name_dislocation') else 'Constant'}")
            if phase.n != -1e23:
                print_ph(f"   n    : {phase.n} ")
            if phase.Edis != -1e23:
                print_ph(f"   Edis : {phase.Edis} ")
            if phase.Vdis != -1e23:
                print_ph(f"   Vdis : {phase.Vdis} ")
            if phase.Bdis != -1e23:
                print_ph(f"   Bdis : {phase.Bdis} ")
            if phase.name_diffusion == 'Constant' and phase.name_dislocation == 'Constant':
                print_ph(f"   eta  : {phase.eta} [Pas] ")    
            
            print_ph('-----------------------------------')
            
            print_ph('-----Thermal Parameters------')
            print_ph(f"Density law       : {phase.name_density if hasattr(phase, 'name_density') else 'Constant'}")
            print_ph(f"Thermal capacity  : {phase.name_capacity if hasattr(phase, 'name_capacity') else 'Constant'}")
            print_ph(f"Thermal conductivity : {phase.name_conductivity if hasattr(phase, 'name_conductivity') else 'Constant'}")
            print_ph(f"Thermal expansivity : {phase.name_alpha if hasattr(phase, 'name_conductivity') else 'Constant'}")
            print_ph(f"Radiogenic heating:  {phase.Hr if phase.Hr !=0.0 else 'Radiogenic heating is not active'}")
            
            if hasattr(phase, 'radio_flag'):
                print_ph(f"   radiative conductivity flag : {phase.radio_flag} ")
            if hasattr(phase, 'rho0'):
                print_ph(f"   rho0 : {phase.rho0} ")
            print_ph('-----------------------------------') 
            if phase.name_capacity == 'Constant':
                print_ph(f"Heat capacity {phase.Cp} J/kg/K")
                print_ph('-----------------------------------')
            if phase.name_conductivity == 'Constant':
                print_ph(f"Thermal conductivity {phase.k} W/m/K")
                print_ph('-----------------------------------') 
            print_ph('\n')
            
            pdb = _generate_phase(pdb,
                                  it, 
                                    radio_flag        = phase.radio_flag if hasattr(phase, 'radio_flag') else 0.0,
                                    rho0              = phase.rho0 if hasattr(phase, 'rho0') else 3300,
                                    name_diffusion    = phase.name_diffusion if hasattr(phase, 'name_diffusion') else 'Constant',
                                    name_dislocation  = phase.name_dislocation if hasattr(phase, 'name_dislocation') else 'Constant',
                                    name_alpha        = phase.name_alpha if hasattr(phase, 'name_alpha') else 'Constant',
                                    name_capacity     = phase.name_capacity if hasattr(phase, 'name_capacity') else 'Constant',
                                    name_density      = phase.name_density if hasattr(phase, 'name_density') else 'Constant',
                                    name_conductivity = phase.name_conductivity if hasattr(phase, 'name_conductivity') else 'Constant',
                                    Edif              = phase.Edif if hasattr(phase, 'Edif') else -1e23,
                                    Vdif              = phase.Vdif if hasattr(phase, 'Vdif') else -1e23,
                                    Bdif              = phase.Bdif if hasattr(phase, 'Bdif') else -1e23,
                                    n                 = phase.n if hasattr(phase, 'n') else -1e23,
                                    Edis              = phase.Edis if hasattr(phase, 'Edis') else -1e23,
                                    Vdis              = phase.Vdis if hasattr(phase, 'Vdis') else -1e23,
                                    Bdis              = phase.Bdis if hasattr(phase, 'Bdis') else -1e23,
                                    eta               = phase.eta if hasattr(phase, 'eta') else 1e20)
            it += 1

    return pdb 

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
    g_input.sub_constant_flag = IP.van_keken
    
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


def StonedFenicsx(IP,Ph_input):
    #---------------------------------------------------------------------------------------------------------
    # Input parameters 
    #---------------------------------------------------------------------------------------------------------
        
    
    # Numerical controls
    ctrl = NumericalControls(g               = IP.g,
                            v_s              = np.asarray(IP.v_s),
                            slab_age         = IP.slab_age,
                            time_max         = IP.time_max,
                            time_dependent_v = IP.time_dependent_v,
                            steady_state     = IP.steady_state,
                            slab_bc          = IP.slab_bc,
                            decoupling       = IP.decoupling_ctrl,
                            tol_innerPic     = IP.tol_innerPic,
                            tol_innerNew     = IP.tol_innerNew,
                            van_keken        = IP.van_keken,
                            van_keken_case   = IP.van_keken_case,
                            model_shear      = dict_options[IP.model_shear],
                            phase_wz         = IP.phase_wz,
                            dt = IP.dt_sim,
                            adiabatic_heating = IP.adiabatic_heating,
                            Tmax=IP.Tmax,
                            it_max=IP.it_max,
                            tol=IP.tol)
    # IO controls
    io_ctrl = IOControls(test_name = IP.test_name,
                        path_save = IP.path_test,
                        sname = IP.sname)
    io_ctrl.generate_io()
    # Scaling parameters
    sc = Scal(L=IP.L,Temp = IP.Temp,eta = IP.eta, stress = IP.stress)
    # LHS parameters
    lhs = ctrl_LHS(nz=IP.nz,
                    end_time = IP.end_time,
                    dt = IP.dt,
                    slab_tk=IP.slab_tk,
                    recalculate = IP.recalculate,
                    van_keken = IP.van_keken,
                    non_linearities=IP.self_consistent_flag,
                    c_age_plate = IP.c_age_plate)
    
    Pdb = generate_phase_database(IP,Ph_input)                      
    # ---
    # Create mesh 
    g_input = fill_geometrical_input(IP)



    

    # Scaling
    ctrl = _scaling_control_parameters(ctrl, sc)
    Pdb = _scaling_material_properties(Pdb,sc)
    lhs = _scale_parameters(lhs, sc)

    M = cm(io_ctrl, sc,g_input,ctrl)
    
    M.element_p = basix.ufl.element("Lagrange","triangle", 1) 
    M.element_PT = basix.ufl.element("Lagrange","triangle",2)
    M.element_V = basix.ufl.element("Lagrange","triangle",2,shape=(2,))
    
    
    solution_routine(M, ctrl, lhs, Pdb, io_ctrl, sc)

    # Create mesh
    return 0    

#---------------------------------------------------------------------------
 
if __name__ == '__main__': 

    StonedFenicsx()   
    
    
        
    
    
    

