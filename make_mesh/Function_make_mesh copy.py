
import numpy as np

import matplotlib.pyplot as plt
# TO DO -> create template classes for points, lines, loop lines, surfaces & physical objects. This would guarantee a certain flexibility and a more organised work flow
# For instance, some function in the writing.geo file are pretty redundant, and can be improved. 

class c_phase():
    def __init__(self,
                 cr=35e3,
                 ocr=6e3,
                 lit_mt=30e3,
                 lc = 0.5,
                 wc = 1.5e3):
        self.cr = cr 
        self.ocr = ocr
        self.lit_mt = lit_mt 
        self.lc = lc
        self.wc = wc 
        self.lt_d = (cr+lit_mt)
        self.decoupling = 100e3
        



def function_create_slab_channel(data_real:bool,c_phase,SP=[],fname=[]): 

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
    cx,cy = function_create_subduction_channel(ax,ay,theta_mean,c_phase)
    ox,oy = function_create_oceanic_crust(ax,ay,theta_mean,c_phase.ocr)
    # Correct the slab surface and find the extra node
    ax,ay,theta_mean = find_extra_node(ax,ay,theta_mean,c_phase)
    # Correct the subduction channel as well finding the extranode 
    # I needed to correct first the slab and then the channel, because otherwise it was
    # creating an unrealistic bending of the weak zone. Initially I was recomputing the angle 
    # between the linear segment, using an average. It is more convinient use two // lines and then 
    # correcting them. 
    cx,cy,ex,ey = correct_channel(cx,cy,ax,ay,c_phase)
    isch = np.where(ay>=-c_phase.lt_d)

    return ax,ay,theta_mean,cx,cy,ex,ey,isch[-1],ox,oy


def function_create_subduction_channel(sx,sy,th,c_phase):
    wc = c_phase.wc
    cx = np.zeros([np.amax(sx.shape),1])
    cy = np.zeros([np.amax(sx.shape),1])
    # Loop over the interface of the slab and find the points on the top of the surface of the subduction channel: the point on the of the top of the channel are perpendicular to the slab interface#
    # Compute the top surface of the subduction channel
    cx = sx + wc*np.sin(th)
    cy = sy + wc*np.cos(th)
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


def function_create_oceanic_crust(sx,sy,th,olt):

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

    
    
def _find_e_node(ax,ay,t,lt,flag=False):
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
    
    return index_extra_node,t_ex


def _correct_(ax,ay,index_extra_node,lt,t,tex):
    
    if index_extra_node == []:
        return ax,ay,t
    
    ax_ex = ax[index_extra_node-1] + ( (ax[index_extra_node] - ax[index_extra_node-1])
                 * (lt - np.abs(ay[index_extra_node-1])) ) / (np.abs(ay[index_extra_node]) - np.abs(ay[index_extra_node-1]))
    
    ax = np.insert(ax,index_extra_node,ax_ex) #ax[index_extra_node-2] + ( (ax[index_extra_node] - ax[index_extra_node-2])* (lt - np.abs(ay[index_extra_node-2])) ) / (np.abs(ay[index_extra_node]) - np.abs(ay[index_extra_node-2])))
    
    ay = np.insert(ay,index_extra_node,-lt)  
    
    t = np.insert(t,index_extra_node,tex)  
    
    return ax,ay,t 
    

def find_extra_node(ax:float,ay:float,t:float,c_phase):

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


def correct_channel(cx,cy,sx,sy,c_phase):
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

def create_subduction_point(sx,sy,lt,dhrs,lc1,lc2,p,pn_):
    """
    function to create the points of the slab
    Input: 
        sx   = coordinate x slab
        sy   = coordinate y slab
        lt   = lithospheric thickness
        dhrs = high resolution depth
        lc1  = resolution 1 
        lc2  = resolution 2
        p    = point list
        pn_  = point number (current)
    Output: 
        p       = update point list
        pn_slab = total number of slab point
        pn_slab_p = important point number for slab [0,interface,bottom]
        pn_     = update number point 
    """
    pn_slab_p = []
    for i in range(len(sx)):
        # Save the important point in the slab. 
        if i == 0:
            pn_slab_p.append(pn_)
        elif sy[i] == -lt:
            pn_slab_p.append(pn_)
            lc = lc1
        elif i  == len(sx)-1:
            pn_slab_p.append(pn_)
        
        if sy[i]>-dhrs:
            lc = lc1
        elif sy[i]<-500e3:
            lc = lc2#*2.2
        else: 
            lc = lc2 
        
        p.append([float(sx[i]),float(sy[i]),0.0,float(lc),int(pn_)])
        pn_ +=1 
    
    pn_slab = pn_-1

    return p,pn_slab,pn_slab_p,pn_

def create_ancor_points(sx,sy,mx,lt,lc_n,lc_lt,lc_tp,p,pn_):
    """
    Function to create the anchor points.
    Input point: 
        sx 
        sy 
        mx 
        lt
        lc_n
        lc_tp
        lc_lt
        p 
        pn_
    Output: 
        p   = update list of points
        pn_ = update number of points
        pac_ = number of point of anchor
        p_ac_p = list of important node
                p_ac_p[0] => left corner
                p_ac_p[1] => right corner (up)
                p_ac_p[2] => intersection lit with right edge
                p_ac_p[3] => right corner (bot)
    """
    pac_ac_p = []
    for i in range(0,4):
        if i == 0:
        # bottom left corner
            a = [sx[0],sy[-1],0.0,lc_n,pn_]  
        elif i==1:
            a = [mx,sy[0],0.0,lc_n,pn_]
        elif i==2:
            a = [mx,-lt,0.0,lc_lt,pn_]
        elif i==3:
            a = [mx,sy[-1],0.0,lc_n,pn_]
        
        pac_ac_p.append(pn_) 
     
        p.append([float(a[0]),float(a[1]),a[2],a[3],int(pn_)])
        pn_ +=1

    pac_ = pn_-1

    return p,pac_,pac_ac_p,pn_

def create_channel_points(cx,cy,dhrs,lt,lc1,lc2,p,ex,ey,pn_):
    """
    Function to create the anchor points.
    Input point: 
        cx 
        cy 
        dhrs 
        lt
        lc_1
        lc2
        ex
        ey
        p 
        pn_
    Output: 
        p   = update list of points
        pn_ = update number of points
        pac_ = number of point of anchor
         p_ac_p = list of important node
            p_ac_p[0] => left corner
            p_ac_p[1] => right corner (up)
            p_ac_p[2] => intersection lit with right edge
            p_ac_p[3] => right corner (bot)
"""
    pch_ch_p = []
    for i in range(len(cx)):
        if cy[i]>-dhrs:
            if i==0:
                a=[cx[i],0.0,0.0,lc1,pn_]
                pch_ch_p.append(pn_)
            else: 
                a = [cx[i],cy[i],0.0,lc2,pn_]

        else:
            a = [cx[i],cy[i],0.0,lc2,pn_]
            if cy[i] == -lt:
                pch_ch_p.append(pn_)
        p.append([float(a[0]),float(a[1]),float(a[2]),float(a[3]),int(a[4])])
        pn_ +=1 
    pch_ch_p.append(pn_-1)
    pch_ = pn_-1
    pch_ch_p.append(pn_)

    p.append([float(ex),float(ey),0.0,lc2,int(pn_)])
    pn_ +=1
    return p,pch_,pch_ch_p,pn_

def create_boundary_line(lines_list,ln_,points_boundary):
    """
    
    
    """

    num_line = len(points_boundary)-1
    list_boundary = []
    for i in range(num_line):
        if i == num_line-1:
            lines_list.append([points_boundary[i],points_boundary[0],ln_])
        else:
            lines_list.append([points_boundary[i],points_boundary[i+1],ln_])
        list_boundary.append(ln_)
        ln_ +=1 

    return lines_list,list_boundary,ln_ 

def create_subudction_lines(lines_list,ln_,point_list,pn_slab,pn_slab_p,d_ch):
    """
    
    
    """
    i_max = pn_slab 
    line_subduction = []
    lit_point = pn_slab_p[1]
    for i in range(i_max-1): 
        p_i = point_list[i][-1]
        p_i1= point_list[i+1][-1]
        lines_list.append([p_i,p_i1,ln_])
        line_subduction.append(ln_)
        if p_i1 == lit_point:
            line_lit_S = i
        if d_ch != 0:
            if p_i1==d_ch:
                line_lit_S2 = i 
        ln_+=1
    
    return lines_list,line_subduction,line_lit_S,line_lit_S2,ln_


def create_channel_lines(lines_list,ln_,point_list,pn_an,pn_ch,pn_ch_p):
    """
    
    
    """
    channel_list = []
    lin_num = pn_ch-pn_an-1
    i_ch = 1
    for i in range(pn_an,pn_ch-1):
        p_i = point_list[i][-1]
        p_i1= point_list[i+1][-1]
        lines_list.append([p_i,p_i1,ln_])
        channel_list.append(ln_)
        if p_i1 == pn_ch_p[1]-1:
            line_lit_C = i_ch
        i_ch = i_ch + 1
        ln_+=1
    
    return lines_list,channel_list,line_lit_C,ln_
def create_lithospheric_lines(lines_list,ln_,point_list,p_b_ovr):
    """
    
    
    
    """
    num_lin = len(p_b_ovr)
    litho_line = []
    for i in range(num_lin-1):
        p_i = point_list[p_b_ovr[i]-1][-1]
        p_i1= point_list[p_b_ovr[i+1]-1][-1]
        lines_list.append([p_i,p_i1,ln_])
        litho_line.append(ln_)
        ln_+=1
    
    return lines_list,litho_line,ln_ 
def create_loop_left(ss,bb,lb,tag):
    """
    
    i know an overkill
    """
    loop_lines = []
    for i in range(len(ss)+2):
        if i == len(ss):
            loop_lines.append(-bb[1])
        elif i == len(ss)+1: 
            loop_lines.append(-lb[0])
        else:
            loop_lines.append(-ss[i])


    left_loop = [[tag],loop_lines]
    return left_loop

def create_loop_channel(cc,ss,tt,ltl,tag):
    """
    
    
    """

    loop_lines = []
    for i in range(len(ss)):
        loop_lines.append(ss[i])
    
    loop_lines.append(-ltl[-1])
    loop_lines.append(-ltl[-2])
    for i in reversed(cc):
        loop_lines.append(-i)
    loop_lines.append(-tt)

    channel_loop = [[tag],loop_lines]

    return channel_loop 

def create_loop_channel2(cc,ss,tt,closing,tag):
    """
    
    
    """

    loop_lines = []
    for i in range(len(ss)):
        loop_lines.append(ss[i])

    loop_lines.append(closing)
    
    for i in reversed(cc):
        loop_lines.append(-i)
    
    
    loop_lines.append(tt[1])
    loop_lines.append(tt[2])

    channel_loop = [[tag],loop_lines]

    return channel_loop 


def create_lithospheric_loop(ll,cc,tt,rb,tag):
    """
    
    """
    line_loop = []

    for i in cc: 
        line_loop.append(i)
    line_loop.append(-ll[0])
    line_loop.append(-rb)
    line_loop.append(-tt) 


    litho_loop = [[tag],line_loop]


    return litho_loop


def create_mantle_loop(ss,lt,bb,rb,tag):
    """
    
    """
    line_loop = []

    for i in range(len(ss)):
        line_loop.append(ss[i])
    
    line_loop.append(-bb)
    line_loop.append(-rb)
    for i in range(len(lt)):
        line_loop.append(lt[i])

    mantle_loop = [[tag],line_loop]


    return mantle_loop
def create_mantle_loop2(ss,ch,cl,lt,bb,rb,tag):
    """
    
    """
    line_loop = []

    for i in range(len(ss)):
        line_loop.append(ss[i])
    
    line_loop.append(-bb)
    line_loop.append(-rb)
    line_loop.append(lt[0])
    for i in range(len(ch)):
        line_loop.append(ch[i])
    line_loop.append(-cl)

    mantle_loop = [[tag],line_loop]


    return mantle_loop
