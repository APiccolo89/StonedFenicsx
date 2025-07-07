# making a .geo file for GMSH, so that we can produce a mesh file (with .msh extension)

import numpy as np
import matplotlib.pyplot as plt
import gmsh as gm
from pylab import figure, axes, pie, title, show
import sys
import os
from solver_function.numerical_control import IOControls
from   make_mesh.Subducting_plate import Slab
import make_mesh.Function_make_mesh as fmm
import make_mesh.writing_geo_file   as wgm 

def _create_mesh_(sname='fake',width_channel=1.5e3,lithosphere_thickness=50e3,depth_high_res_trench=1.5e3,lc_normal=10e3,lc_top = 10.0e3,lc_litho=15.0e3,Data_Real=False,S=[],real_slab_file=[],X=[0,1e6],Z=[-660e3,0],path_test='',van_keken = 1):
    """
    Input: 
        width_channel          = the width of the subduction channel [km]
        lithosphere thickness  = lithosphere thickness [km]
        lc_normal              = resolution outside the area of reference [km]
        lc_lithosphere         = resolution within the lithosphere [km] 
        lc_corner              = resolution in the left corner [km] 
        data_real              = Flag (do we use a file.txt with the slab data?)
        Slab()                 = Slab class filled with the information required to create a setup without real data (optional)
        real_slab_file         = file with the slab data in txt
    Output: 
        mesh file that is read by iFieldStone 

    Short explanation: 
        This script, simply create a mesh using either the generation of a .geo file {I will update the script}, or the embedded geometries 
        1st -> Generate the points with the geometry [P1,2,3,4] are the point defining the boundary 
        2nd -> Generate the line folowing the main unit in the setup: 
               [Slab line ]       -> collection of lines that are defined by the slab points [LN2aS-LNbS]
               [Channel line]     -> collection of lines that define the top surface of the subduction channel[LNaC-LNbC] 
               [Base lithosphere] -> collection of lines that define the base of the overridding plate
                                    1. line from the right boundary to the channel intersection with lithosphere [LNaC+1]
                                    2. line from the channel intersection to the middle point of the channel at this depth [LNaC+2]
                                    3. line from the mid point to the subduction top surface intersection with the slab top surface [LNaC+3]
               [Top Boundary]     -> collection of lines that defines the top boundary: 
                                    1. line from the first point of the slab to the intersection of the channel with the "surface" [L1]
                                    2. line from the intersection of channel into the surface to the P2 [L2]
               [Right Boundary]   -> collections of lines that define the right boundary 
                                    1. line from the P2 to the depth of the lithosphere 
                                    2. line from the depth of lithosphere to P3 
               [Bottom Boundary]  -> collections of lines that define the bottom boundary
                                    1. line from P3 and the point of intersection of the slab with the bottom boundary
                                    2. line from the intersection of the slab to the P4 
               [Left Boundary]    -> collections of line that define the left boundary: 
                                    1. the line between P4-P1
                => At the the end generate the physical line group
        3rd -> Line loops that define 4 region of the numerical models: 
                [loop_S]       -> line loop that defines the left area of the model 
                [channel_loop] -> line loop that defines the channel 
                [OVP]          -> line loop that defines the overriding plate
                [right_loop]   -> line loop that define the area on the right side and below the overriding plate
        4th -> Generate the surface that are defined by the loop
        5th -> Generate the physical surface
        6th -> generate the mesh. 
    """
    min_x           = X[0] # The beginning of the model is the trench of the slab
    max_x           = X[1]                 # Max domain x direction
    min_y           = Z[0] # Min domain y direction
    max_y           = Z[1] # Max domain y direction 
    
    lc_interface = width_channel     # The resolution along the subduction zone
    lc_left_interface = lc_interface # The interface 

    print('========= Initialising subduction surface using: ')

    if (Data_Real==False):
            print('              : Slab class')
    else: 
            print('              : Real slab (%s)')
    
  
    # Check if exist real data, and if subduction data are empty for the default function 
    if (Data_Real==False) & (isinstance(S, Slab)== False):
        if van_keken == 1: 
            min_x           = 0.0 # The beginning of the model is the trench of the slab
            max_x           = 660e3                 # Max domain x direction
            min_y           = -600e3 # Min domain y direction
            max_y           = 0.0 # Max domain y direction 
            S = Slab(D0 = 100.0, L0 = 800.0, Lb = 400, trench = 0.0,theta_0=5, theta_max = 45.0, num_segment=100,flag_constant_theta=True,y_min=min_y)
        else: 
            S = Slab(D0 = 100.0, L0 = 800.0, Lb = 400, trench = 0.0,theta_0=1.0, theta_max = 45.0, num_segment=100,flag_constant_theta=False,y_min=min_y)

        for a in dir(S):
            if (not a.startswith('__')):
                att = getattr(S, a)
                if (callable(att)==False) & (np.isscalar(att)):
                    print('%s = %.2f'%(a, att))

    # Create the subduction interfaces using either the real data set, or the slab class
    slab_x, slab_y, theta_mean,channel_x,channel_y,extra_x,extra_y,isch = fmm.function_create_slab_channel(Data_Real,width_channel,lithosphere_thickness,S,real_slab_file)
    print('========= Finished =============== ')
    #print('===============Numerical domain==================')
    print('x                               = [%.2f,%.2f] [m]' %(min_x,max_x))
    print('y                               = [%.2f,%.2f] [m]' %(min_y,max_y))
    print('Lithosphere thickness           = %.2f [m]' %(lithosphere_thickness))
    print('Depth high resolution           = %.2f [m]'%(depth_high_res_trench))
    print('==')
    print('Resolution left corner          = %.2f [m]' %lc_left_interface)
    print('Resolution subduction interface = %.2f [m]' %lc_interface)
    print('Resolution lithopshere          = %.2f [m]' %lc_litho)
    print('Resolution top                  = %.2f [m]' %lc_top)
    print('Resolution normal               = %.2f [m]' %lc_normal)
    print('=================================================')
    print('===============Creating points===============================================================')
    # creating slab interface points 
    point_list = [] # empty list -> this list contains the information of the point created
                     # p([px,py,pz,lc,tag])
                     # px,py,pz => coordinate (pz == 0.0) [m]
                     # lc       => resolution 
                     # tag      => unique number identification (natural number)
    pn_  = 1 # point number list (point list -> pn_-1 -> point(x,y,z...pn_))
    
    point_list,pn_slab,pn_slap_p,pn_ = fmm.create_subduction_point(slab_x,slab_y,lithosphere_thickness,depth_high_res_trench,lc_left_interface,lc_interface,point_list,pn_)
    print('============= number points of the slab int = %d'%(pn_-1))

    point_list,pn_an,pn_an_p,pn_ = fmm.create_ancor_points(slab_x,slab_y,max_x,lithosphere_thickness,lc_normal,lc_litho,lc_top,point_list,pn_)
    print('============= number anchor points          = %d'%(pn_an-pn_slab))

    point_list,pn_ch,pn_ch_p,pn_ = fmm.create_channel_points(channel_x,channel_y,depth_high_res_trench,lithosphere_thickness,lc_left_interface,lc_interface,point_list,extra_x,extra_y,pn_)
    print('============= number channel points         = %d'%(pn_ch-pn_an))

    print('============= extra node                    = %d'%(1))

    print('============= total number points           = %d'%(pn_-1))
    print('======================== Finished ========================================================== ')

    print('=============== Creating lines ==============================================================')
    lines_list = []
    ln_ = 1 
    # point boundary:
    # The top boundary has two segment (from left to right): from point 1 to the intersection of the channel in the surface; from the intersection of the channel to the right top boundary
    # The right boundary has two segment(from top to bottom): from the top right corner to the lithospheric thickness; from the lithospheric thickness to the bottom right boundary
    # The bottom boudnary has two segment (from right to left): from the bottom right corner to the intersection of the slab with bottom boundary; from the bottom intersection of the slab to the left corner boundary 
    # The left boundary (from bottom to top) has only one segment from the left bottom corner boundary to the first slab node
    points_boundary = [pn_slap_p[0], # 1st point slab (left top boundary)
                       pn_ch_p[0],   # 1st point channel (intersection surf)
                       pn_an_p[1],   # left right boundary 
                       pn_an_p[2],   # intersection lithosphere with edge
                       pn_an_p[3],   # left bottom boundary
                       pn_slap_p[-1],# intersection slab bottom boundary
                       pn_an_p[0],   # right bottom boundary
                       pn_slap_p[0]]
    lines_list,boundary_lines,ln_ = fmm.create_boundary_line(lines_list,ln_,points_boundary)
    
    print('============= Boundary lines performed      = %d'%(len(lines_list)))

    lines_list,subduction_lines,index_lithos,index_lithos2,ln_= fmm.create_subudction_lines(lines_list,ln_,point_list,pn_slab,pn_slap_p,isch[-1]+2)

    print('============= Subduction lines performed    = %d'%(len(subduction_lines)))

    lines_list,channel_line,index_lithosC,ln_= fmm.create_channel_lines(lines_list,ln_,point_list,pn_an,pn_ch,pn_ch_p)

    print('============= Channel lines performed       = %d'%(len(channel_line)))

    points_base_overriding = [pn_an_p[2],     #intersection lithosphere and right boundary
                              pn_ch_p[1],   #intersection lithosphere and channel
                              pn_ch_p[3],   # middle point at lithosphere depth (Between channel and subduction)
                              pn_slap_p[1]] # intersection slab and lithospheric depthb

    lines_list,lithospheric_line,ln_ = fmm.create_lithospheric_lines(lines_list,ln_,point_list,points_base_overriding)
    
    p_i  = isch[-1]+1
    p_i1 = pn_ch_p[2] 
    
    close_channel_line = ln_
    lines_list.append([p_i,p_i1,ln_])


    print('============= Lithospheric lines performed  = %d'%(len(lithospheric_line)))
    print('============= Total lines performed         = %d'%((ln_)-1))
    print('======================== Finished ========================================================== ')
    print('=============== Creating physical line =====================================================')
    Top_boundary        = [[111],[boundary_lines[0],boundary_lines[1]]]
    Right_Boundary      = [[112],[boundary_lines[2],boundary_lines[3]]]
    Bottom_Boundary     = [[113],[boundary_lines[4],boundary_lines[5]]]
    Left_Boundary       = [[114],[boundary_lines[6]]]
    Slab_Surface        = [[101],subduction_lines]
    Channel_Surface     = [[102],channel_line]
    Overriding_plate_v0 = [[104],[lithospheric_line[0]]]
    Overriding_plate_v1 = [[105],[lithospheric_line[1],lithospheric_line[2]]]
    print('======================== Finished ========================================================== ')
    print('=============== Creating loop line =========================================================')
    # For meshing, you need to create a loop: subduction_line, left line of the bottom, then left boundary
    Left_side_loop = fmm.create_loop_left(subduction_lines,Bottom_Boundary[1][:],Left_Boundary[1][:],10)
    Channel_loop = fmm.create_loop_channel(channel_line[0:index_lithosC+1],subduction_lines[0:index_lithos+1],Top_boundary[1][0],lithospheric_line,20)

    Mantle_above = fmm.create_mantle_loop2(subduction_lines[(index_lithos2):],channel_line[index_lithosC+1:],close_channel_line,lithospheric_line,Bottom_Boundary[1][0],Right_Boundary[1][1],30)
    Lithosphere_loop = fmm.create_lithospheric_loop(lithospheric_line,channel_line[0:index_lithosC+1],Top_boundary[1][1],Right_Boundary[1][0],40)
    Channel_loop2 = fmm.create_loop_channel2(channel_line[index_lithosC+1:],subduction_lines[index_lithos+1:index_lithos2],lithospheric_line,close_channel_line,50)

    # Writing the geo file
    wgm.writing_geo(sname
                    ,point_list
                    ,pn_slab
                    ,pn_an
                    ,pn_ch
                    ,lines_list
                    ,boundary_lines
                    ,subduction_lines
                    ,channel_line
                    ,lithospheric_line
                    ,Left_side_loop
                    ,Channel_loop
                    ,Mantle_above
                    ,Lithosphere_loop
                    ,Top_boundary
                    ,Right_Boundary
                    ,Bottom_Boundary
                    ,Left_Boundary
                    ,Slab_Surface
                    ,Channel_Surface
                    ,Overriding_plate_v0
                    ,Overriding_plate_v1,
                    Channel_loop2
                    ,path_test
                    )
    print(sname)
    geo_file = '%s.geo'%sname
    geo_file = os.path.join(path_test,geo_file)
    msh_name = os.path.join(path_test,'%s.msh'%sname)
    gm.initialize()
    gm.open(geo_file)
    gm.model.geo.synchronize()
    gm.model.mesh.generate(2)
    gm.model.mesh.setOrder(2)
    gm.write(msh_name)
    gm.finalize()

    print('done')



"""
Useful stuff for debugging. I should create a set of test functions
for i in range(len(channel_line)):
   p0 = lines_list[channel_line[i]-1][0]
   p1 = lines_list[channel_line[i]-1][1]
   plt.scatter(point_list[p0-1][0],point_list[p0-1][1])
   plt.scatter(point_list[p1-1][0],point_list[p1-1][1])



"""