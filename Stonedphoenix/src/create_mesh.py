# input for iFieldstone 

import os 
import sys
sys.path.append(os.path.abspath("src"))
import Function_make_mesh 
from Function_make_mesh import Class_Points
from Function_make_mesh import Class_Line

import numpy as np
import gmsh 

# dictionary for surface and phase. 
"""
 Long comment: physical surfaces are geometrical subdomain where I solve a set of equation/define phases. The physical surface, in this context,
 are the subdomain of the phases.
 -> sub_plate is the inflowing mantle
 -> oceanic crust .. 
"""

class geom_input():
    def __init__(self,
                 x = [0.,1000e3],
                 y = [-660e3,0.],
                 cr=35e3,
                 ocr=6e3,
                 lit_mt=30e3,
                 lc = 0.5,
                 wc = 1.5e3,
                 decoupling = 100e3):
        self.x  = x                    # main grid coordinate
        self.y  = y   
        self.cr = cr                   # crust 
        self.ocr = ocr                 # oceanic crust
        self.lit_mt = lit_mt           # lithosperic mantle  
        self.lc = lc                   # lower crust ratio 
        self.wc = wc                   # weak zone 
        self.lt_d = (cr+lit_mt)        # total lithosphere thickness
        self.decoupling = decoupling   # decoupling depth -> i.e. where the weak zone is prolonged 


class Mesh(): 
    def __init__(self):
        self.meshv_o  
        self.mesht_o   
        self.P        
        self.VP       
        self.PT      
        self.name

       
class class_line():
    pass 




dict_surf = {
    'sub_plate'         : 1,
    'oceanic_crust'     : 2,
    'wedge'             : 3,
    'overriding_lm'     : 4,
    'lower_crust'       : 5,
    'upper_crust'       : 6,
    'Channel_surf_a'    : 7,
    'Channel_surf_b'    : 8,
}

dict_tag_lines = {
    'Top'               : 1,
    'Right'             : 2,
    'Bottom'            : 3,
    'Left'              : 4,
    'Subduction'        : 5,
    'Channel'           : 6,
    'Oceanic'           : 7,
    'Channel_over'      : 8,
    'Overriding_mantle' : 9,
    'Channel_decoupling': 10,
    'Crust_overplate'   : 11,
    'LCrust_overplate'  : 12,
}




def create_gmsh(sx,        # subduction x
                sy,        # subdcution y 
                th,        # theta
                chx,       # channel x
                chy,       # channel y 
                oc_cx,     # oceanic cx 
                oc_cy,     # oceanic cu
                g_input):  # geometry input class 
    
    # -> USE GMSH FUNCTION 
    gmsh.initialize()


    mesh_model = gmsh.model()

    CP = Class_Points()
    mesh_model = CP.update_points(mesh_model,sx,sy,chx,chy,oc_cy,oc_cy,g_input)

    LC = Class_Line()
    mesh_model = LC.update_lines(mesh_model,CP, g_input)


    # Function create Physical line
    #-- Create Physical Line
    

 
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}
    
    mesh_model.addPhysicalGroup(1, tag_L_T, tag=dict_tag_lines['Top'])
    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, tag_L_R, tag=dict_tag_lines['Right'])


    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, tag_L_B, tag=dict_tag_lines['Bottom'])

    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, tag_L_L, tag=dict_tag_lines['Left'])

    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, tag_L_sub,tag=dict_tag_lines['Subduction'])

    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, tag_L_ch,tag=dict_tag_lines['Channel'])
    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, tag_L_oc, tag=dict_tag_lines['Oceanic'])
  
    
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, tag_L_ch_ov, tag=dict_tag_lines['Channel_over']) 

    
    gmsh.model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.addPhysicalGroup(1, tag_L_ov, tag=dict_tag_lines['Overriding_mantle'])


    if g_input.cr !=0: 

        mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

        mesh_model.addPhysicalGroup(1, tag_L_cr, tag=dict_tag_lines['Crust_overplate'])



        if g_input.lc !=0:
            mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}
            mesh_model.addPhysicalGroup(1, tag_L_Lcr,tag=dict_tag_lines['LCrust_overplate'])
   

    # Left side of the subduction zone 10 
    # Oceanic crust 15
    # Right side of the subduction zone 20
    # Right mantle 25
    # Lithospheric mantle 25
    # Crust LC  30
    # Crust UC  35 
    # Channel 
    
    # left side of the subduction
    a = []
    a.extend(lines_oc[2,:])
    a.extend([lines_B[2,-1]])
    a.extend([-lines_L[2,0]])
    for i in range(len(a)):
        l = np.abs(a[i])
        i_l = np.where(line_global[2,:]==l)
        p0  = line_global[0,i_l][0][0]
        p1  = line_global[1,i_l][0][0]
        p_i_0 = np.where(global_points[2,:]==p0)
        p_i_1 = np.where(global_points[2,:]==p1)
        x   = [global_points[0,p_i_0][0],
               global_points[0,p_i_1][0]]
        y   = [global_points[1,p_i_0][0],
               global_points[1,p_i_1][0]]
        plt.plot(x,y,c='k')
    
    
    loop1 = mesh_model.geo.addCurveLoop(a,10) # Left side of the subudction zone

    # oceanic crust 
    a = []
    a.extend(lines_S[2,:])
    a.extend([lines_B[2,1]])
    b = (lines_oc[2,:])*-1
    a.extend(b[::-1])
    a.extend([-lines_L[2,-1]])
    for i in range(len(a)):
        l = np.abs(a[i])
        i_l = np.where(line_global[2,:]==l)
        p0  = line_global[0,i_l][0][0]
        p1  = line_global[1,i_l][0][0]
        p_i_0 = np.where(global_points[2,:]==p0)
        p_i_1 = np.where(global_points[2,:]==p1)
        x   = [global_points[0,p_i_0][0],
               global_points[0,p_i_1][0]]
        y   = [global_points[1,p_i_0][0],
               global_points[1,p_i_1][0]]
        plt.plot(x,y,c='b')
   
    
    
    
    loop2 = mesh_model.geo.addCurveLoop(a,15) # Oceanic crust 
    
    # right_mantle 
    # From p0 belonging ch -> ch->base channel -> subduction -> 
    
    a = []
    a.extend([-lines_R[2,-1]])
    a.extend([-lines_B[2,0]])
    # find index subduction 
    'Select the node of the slab from bottom to the base of decoupling'
    index = find_line_index(lines_S,coord_sub,g_input.decoupling)
    index = index 
    buf_array = lines_S[2,index:]
    chose_sub = -buf_array 
    chose_sub = chose_sub[::-1]
    a.extend(chose_sub)
    a.extend(lines_base_ch[2,:])
    # Make a small function out of it. 
    
    index = find_line_index(lines_ch,coord_channel,g_input.lt_d)
    index = index 
    choose_sub = []
    buf_array = np.array(lines_ch[2,index:])
    chose_sub = -1*buf_array 
    chose_sub = chose_sub[::-1]
    a.extend(chose_sub)
    a.extend(lines_L_ov[2,:])

    for i in range(len(a)):
        l = np.abs(a[i])
        i_l = np.where(line_global[2,:]==l)
        p0  = line_global[0,i_l][0][0]
        p1  = line_global[1,i_l][0][0]
        p_i_0 = np.where(global_points[2,:]==p0)
        p_i_1 = np.where(global_points[2,:]==p1)
        x   = [global_points[0,p_i_0][0],
               global_points[0,p_i_1][0]]
        y   = [global_points[1,p_i_0][0],
               global_points[1,p_i_1][0]]
        plt.plot(x,y,c='r')
    


    loop2 = mesh_model.geo.addCurveLoop(a,20) # Right mantle 

    if g_input.cr !=0:
            
        index_a    = find_line_index(lines_ch,coord_channel,g_input.cr)
        index_b    = find_line_index(lines_ch,coord_channel,g_input.lt_d)-1
        buf_array = lines_ch[2,index_a:index_b+1]
        buf_array = -buf_array[::-1]
        a = []

        a.extend([-lines_R[2,2]])
        a.extend([-lines_L_ov[2,:].item()])
        a.extend(buf_array)
        a.extend([lines_cr[2,:][0]])


        
        for i in range(len(a)):
            l = np.abs(a[i])
            i_l = np.where(line_global[2,:]==l)
            p0  = line_global[0,i_l][0][0]
            p1  = line_global[1,i_l][0][0]
            p_i_0 = np.where(global_points[2,:]==p0)
            p_i_1 = np.where(global_points[2,:]==p1)
            x   = [global_points[0,p_i_0][0],
                   global_points[0,p_i_1][0]]
            y   = [global_points[1,p_i_0][0],
                   global_points[1,p_i_1][0]]
            plt.plot(x,y,c='g')
    
        # Permanent subcrustal mantle 
        mesh_model.geo.addCurveLoop(a,25) # Right mantle
        
        if g_input.lc !=0:
            a = []
            index_a    = find_line_index(lines_ch,coord_channel,(1-g_input.lc)*g_input.cr)
            index_b    = find_line_index(lines_ch,coord_channel,g_input.cr)-1

            buf_array = -lines_ch[2,index_a:index_b+1]
            buf_array = buf_array[::-1]
            
            a = []

            a.extend([-lines_R[2,1]])
            a.extend([-lines_cr[2,:].item()])
            a.extend(buf_array)
            a.extend([lines_lcr[2,:].item()])

            
            for i in range(len(a)):
                l = np.abs(a[i])
                i_l = np.where(line_global[2,:]==l)
                p0  = line_global[0,i_l][0][0]
                p1  = line_global[1,i_l][0][0]
                p_i_0 = np.where(global_points[2,:]==p0)
                p_i_1 = np.where(global_points[2,:]==p1)
                x   = [global_points[0,p_i_0][0],
                       global_points[0,p_i_1][0]]
                y   = [global_points[1,p_i_0][0],
                       global_points[1,p_i_1][0]]
                plt.plot(x,y,c='firebrick')
    


            mesh_model.geo.addCurveLoop(a,30)
 
            a = []
            index_a    = 0
            index_b    = find_line_index(lines_ch,coord_channel,(1-g_input.lc)*g_input.cr)-1

            buf_array = -lines_ch[2,index_a:index_b+1]
            buf_array = buf_array[::-1]
            
            a.extend([lines_R[2,0]])
            a.extend([-lines_lcr[2,:].item()])
            a.extend(buf_array)
            a.extend([lines_T[2,1]])

            
            for i in range(len(a)):
                l = np.abs(a[i])
                i_l = np.where(line_global[2,:]==l)
                p0  = line_global[0,i_l][0][0]
                p1  = line_global[1,i_l][0][0]
                p_i_0 = np.where(global_points[2,:]==p0)
                p_i_1 = np.where(global_points[2,:]==p1)
                x   = [global_points[0,p_i_0][0],
                       global_points[0,p_i_1][0]]
                y   = [global_points[1,p_i_0][0],
                       global_points[1,p_i_1][0]]
                plt.plot(x,y,c='r')

            
            mesh_model.geo.addCurveLoop(a,35)

    a = []
    index_a = find_line_index(lines_ch,coord_channel,g_input.lt_d)
    index = find_line_index(lines_S,coord_sub,g_input.lt_d)

    buf_array = -lines_S[2,0:index]
    buf_array = buf_array[::-1]
    a.extend(np.int32(lines_ch[2,0:index_a]))
    a.extend([-lines_ch_ov[2,:].item()])
    a.extend(buf_array)
    a.extend([lines_T[2,0]])
    mesh_model.geo.addCurveLoop(a,40)





    a = []
    index_a = find_line_index(lines_S,coord_sub,g_input.lt_d)
    index = find_line_index(lines_S,coord_sub,g_input.decoupling)
    buf_array = -lines_S[2,index_a:index]
    buf_array = buf_array[::-1]
    index_a = find_line_index(lines_ch,coord_channel,g_input.lt_d)
    a.extend(np.int32(lines_ch[2,index_a:]))
    a.extend([-lines_base_ch[2,:].item()])
    a.extend(buf_array)
    a.extend([lines_ch_ov[2,:].item()])
    for i in range(len(a)):
        l = np.abs(a[i])
        i_l = np.where(line_global[2,:]==l)
        p0  = line_global[0,i_l][0][0]
        p1  = line_global[1,i_l][0][0]
        p_i_0 = np.where(global_points[2,:]==p0)
        p_i_1 = np.where(global_points[2,:]==p1)
        x   = [global_points[0,p_i_0][0],
               global_points[0,p_i_1][0]]
        y   = [global_points[1,p_i_0][0],
               global_points[1,p_i_1][0]]
        plt.plot(x,y,c='k')
    
    mesh_model.geo.addCurveLoop(a,45)

    
    Left_side_of_subduction_surf   = gmsh.model.geo.addPlaneSurface([10],tag=100) # Left side of the subudction zone
    Oceanic_Crust_surf             = gmsh.model.geo.addPlaneSurface([15],tag=150) # Left side of the subudction zone
    Right_side_of_subduction_surf  = gmsh.model.geo.addPlaneSurface([20],tag=200) # Right side of the subudction zone    
    Lithhospheric_Mantle_surf      = gmsh.model.geo.addPlaneSurface([25],tag=250) # Right mantle
    Crust_LC_surf                  = gmsh.model.geo.addPlaneSurface([30],tag=300) # Crust LC
    Crust_UC_surf                  = gmsh.model.geo.addPlaneSurface([35],tag=350) # Crust LC
    Channel_surf_A                 = gmsh.model.geo.addPlaneSurface([40],tag=400) # Channel
    Channel_surf_B                 = gmsh.model.geo.addPlaneSurface([45],tag=450) # Channel

    
    mesh_model.geo.synchronize()

    mesh_model.addPhysicalGroup(2, [Left_side_of_subduction_surf], tag=dict_tag_surf['sub_plate'])
    mesh_model.addPhysicalGroup(2, [Oceanic_Crust_surf], tag=dict_tag_surf['oceanic_crust'])
    mesh_model.addPhysicalGroup(2, [Right_side_of_subduction_surf], tag=dict_tag_surf['wedge'])
    mesh_model.addPhysicalGroup(2, [Lithhospheric_Mantle_surf], tag=dict_tag_surf['overriding_lm'])
    mesh_model.addPhysicalGroup(2, [Crust_LC_surf], tag=dict_tag_surf['lower_crust'])
    mesh_model.addPhysicalGroup(2, [Crust_UC_surf], tag=dict_tag_surf['upper_crust'])
    mesh_model.addPhysicalGroup(2, [Channel_surf_A], tag=dict_tag_surf['Channel_surf_a'])
    mesh_model.addPhysicalGroup(2, [Channel_surf_B], tag=dict_tag_surf['Channel_surf_b'])

    
    
    
    return model_gmsh 


# First draft -> I Need to make it a decent function, otherwise this is a fucking nightmare
def create_parallel_mesh(ctrlio):
    """_summary_: The function is composed by three part: -> create points, create lines, loop 
    ->-> 
    Args:
        ctrl (_type_): _description_
        ctrlio (_type_): _description_

    Returns:
        _type_: _description_
    """
    

    import gmsh 
    from   Subducting_plate import Slab
    import Function_make_mesh as fmm 


    g_input = geom_input()
    min_x           = g_input.x[0] # The beginning of the model is the trench of the slab
    max_x           = g_input.x[1]                 # Max domain x direction
    max_y           = g_input.y[1] 
    min_y           = g_input.y[0]                # Min domain y direction
    # Set up slab top surface a
    Data_Real = False; S = []
    van_keken = 0
    if (Data_Real==False) & (isinstance(S, Slab)== False):
        if van_keken == 1: 
            S = Slab(D0 = 100.0, L0 = 800.0, Lb = 400, trench = 0.0,theta_0=5, theta_max = 45.0, num_segment=100,flag_constant_theta=True,y_min=min_y)
        else: 
            S = Slab(D0 = 100.0, L0 = 800.0, Lb = 400, trench = 0.0,theta_0=1.0, theta_max = 45.0, num_segment=100,flag_constant_theta=False,y_min=min_y)

        for a in dir(S):
            if (not a.startswith('__')):
                att = getattr(S, a)
                if (callable(att)==False) & (np.isscalar(att)):
                    print('%s = %.2f'%(a, att))

    # Create the subduction interfaces using either the real data set, or the slab class
    slab_x, slab_y, theta_mean,channel_x,channel_y,extra_x,extra_y,isch,oc_cx,oc_cy = fmm.function_create_slab_channel(Data_Real,g_input,SP=S)


    mesh_model = create_gmsh(slab_x,slab_y,theta_mean,channel_x,channel_y,oc_cx,oc_cy,g_input) 




    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}

    mesh_model.geo.mesh.setAlgorithm(2, 40000, 3)
    mesh_model.geo.mesh.setAlgorithm(2, 45000, 3)
    mesh_model.geo.synchronize()  # synchronize before adding physical groups {thanks chatgpt}


    mesh_model.mesh.generate(2)
    mesh_model.mesh.setOrder(2)
    gmsh.write("experimental.msh")
    
    return 0
    



def unit_test():
    import numpy as np 
    import sys, os
    sys.path.append(os.path.abspath("src"))
    from numerical_control import IOControls

    
    
    g_input = geom_input() 
    
    IOCtrl = IOControls(test_name = 'Debug_test',
                        path_save = '../Debug_mesh',
                        sname = 'Experimental')
    IOCtrl.generate_io()
    create_parallel_mesh(IOCtrl)
    
    
    
    assert Passed == True 
    
    

if __name__ == '__main__':
    
    unit_test()


    

