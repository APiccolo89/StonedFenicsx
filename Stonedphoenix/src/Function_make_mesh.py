
import numpy as np

import matplotlib.pyplot as plt
# TO DO -> create template classes for points, lines, loop lines, surfaces & physical objects. This would guarantee a certain flexibility and a more organised work flow
# For instance, some function in the writing.geo file are pretty redundant, and can be improved. 


def create_loop(l_list,mesh_model,tag):
    a = []
    
    for i in range(len(l_list)):
        
        if isinstance(l_list[i], (int, np.integer)): 
            val = [l_list[i]]
        else: 
            val = l_list[i]
    
        a.extend(val)
        
    
    mesh_model.geo.addCurveLoop(a,tag) # Left side of the subudction zone
   
    return mesh_model


def find_line_index(Lin_ar,point,d):
    
    for i in range(len(Lin_ar[0,:])-1):
        # Select pint of the given line
        p0 = np.int32(Lin_ar[0,i])

        p1 = np.int32(Lin_ar[1,i])
        # find the index of the point 
        ip0 = np.where(p0==np.int32(point[2,:]))
        ip = np.where(p1==np.int32(point[2,:]))
        print(i,p0,p1)
        # Check wheter or not the coordinate z is the one. 
        z1 = point[1,ip]
        if z1 == -d: 
            X = [point[0,ip0],point[0,ip]]
            Y = [point[1,ip0],point[1,ip]]
            print(X)
            print(Y)
            index = i+1 
            break    
    
    
    return index 


def find_tag_line(coord,x,dir):
    if dir == 'x':
        a = 0 
    else:
        a = 1 
    
    i = np.where(coord[a,:]==-x)
    i = i[0][0];i = coord[2,i]
    
    return np.int32(i)                 


class Class_Points():
    def update_points      (self,
                            mesh_model,
                            sx,
                            sy,
                            cx,
                            cy,
                            oc_cx,
                            oc_cy,
                            g_input):
        """
        To Do Friday: document this function



        """

        min_x   = g_input.x[0]
        min_y   = g_input.y[0]
        max_x   = g_input.x[1]
        max_y   = g_input.y[1]
        # Function CREATE POINTS

        #-- Create the subduction,channel and oceanic crust points 
        # Point of subduction -> In general, the good wisdom would tell you to not put a disordered list of points, 
        # but, I am paranoid, therefore: just create the geometry of slab, channel and oceanic crust such that y = f(x) where f(x) is always growing (or decreasing)
        # if you want to be creative, you have to modify the function of create points, up to you, no one is against it. 

        self.max_tag_s,  self.tag_subduction, self.coord_sub,     mesh_model     = _create_points(mesh_model, sx,    sy,    g_input.wc, 0)
        self.max_tag_c,  self.tag_channel,    self.coord_channel, mesh_model     = _create_points(mesh_model, cx,    cy,    g_input.wc, self.max_tag_s)
        self.max_tag_oc, self.tag_oc,         self.coord_ocean,   mesh_model     = _create_points(mesh_model, oc_cx, oc_cy, g_input.wc, self.max_tag_c)
        # -- Here are the points at the boundary of the model. The size of the model is defined earlier, and subduction zone is modified as such to comply the main geometrical input, 
        # I used subduction points because they define a few important point. 


        self.max_tag_a, self.tag_left_c,    self.coord_lc,  mesh_model            = _create_points(mesh_model,  min_x, min_y,         g_input.wc*2,  self.max_tag_oc, True)
        self.max_tag_b, self.tag_right_c_b, self.coord_bc,  mesh_model            = _create_points(mesh_model,  max_x, min_y,         g_input.wc*2,  self.max_tag_a,  True)
        self.max_tag_c, self.tag_right_c_l, self.coord_lr,  mesh_model            = _create_points(mesh_model,  max_x, -g_input.lt_d, g_input.wc*2,  self.max_tag_b,  True)
        self.max_tag_d, self.tag_right_c_t, self.coord_top, mesh_model            = _create_points(mesh_model,  max_x, max_y,         g_input.wc*2,  self.max_tag_c,  True)

        if g_input.cr !=0: 
            self.max_tag_e, self.tag_right_c_cr, self.coord_crust,    mesh_model  = _create_points(mesh_model, max_x, -g_input.cr,                g_input.wc*2, self.max_tag_d, True)
            if g_input.lc !=0: 
                self.max_tag_f, self.tag_right_c_lcr,  self.coord_lcr, mesh_model = _create_points(mesh_model, max_x, -g_input.cr*(1-g_input.lc), g_input.wc*2, self.max_tag_e, True)

        self.global_points = np.hstack([self.coord_sub, self.coord_channel, self.coord_ocean, self.coord_lc, self.coord_bc, self.coord_lr, self.coord_top, self.coord_crust, self.coord_lcr])

        return mesh_model
 
class Class_Line():
    def update_lines(self, mesh_model, CP, g_input):
        
        # Top Boundary      
        p_list                                                    = [CP.tag_subduction[0],   CP.tag_channel[0],  CP.tag_right_c_t[0]]
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
        p_list                                                      = [CP.tag_right_c_b[0], CP.tag_subduction[-1], CP.tag_oc[-1], CP.tag_left_c[0]]
        self.max_tag_bottom, self.tag_L_B, self.lines_B, mesh_model = _create_lines(mesh_model, self.max_tag_right, p_list, False)
        
        # ] 
        p_list                                                    = [CP.tag_left_c[0], CP.tag_oc[0], CP.tag_subduction[0]]
        self.max_tag_left, self.tag_L_L, self.lines_L, mesh_model = _create_lines(mesh_model,  self.max_tag_bottom,  p_list,  False)
        
        
        # -- Create Lines 
        self.max_tag_line_subduction, self.tag_L_sub, self.lines_S,  mesh_model        = _create_lines(mesh_model,  self.max_tag_left,           CP.tag_subduction,False)
        self.max_tag_line_channel,    self.tag_L_ch,  self.lines_ch, mesh_model        = _create_lines(mesh_model,  self.max_tag_line_subduction,CP.tag_channel,   False)
        self.max_tag_line_ocean,      self.tag_L_oc,  self.lines_oc, mesh_model        = _create_lines(mesh_model,  self.max_tag_line_channel,   CP.tag_oc,        False)
        # Create line overriding plate:
        #-- find tag of the of the channel # -> find the mistake: confusion with index types
        i_s = find_tag_line(CP.coord_sub,     g_input.lt_d,'y')

        i_c = find_tag_line(CP.coord_channel, g_input.lt_d,'y')
        # CHECK!
        p_list = [i_s,i_c]
        self.max_tag_line_ch_ov, self.tag_L_ch_ov, self.lines_ch_ov, mesh_model        = _create_lines(mesh_model,  self.max_tag_line_ocean,  p_list,  False)

        p_list = [i_c, CP.tag_right_c_l[0]]
        self.max_tag_line_ov,    self.tag_L_ov,    self.lines_L_ov,  mesh_model        = _create_lines(mesh_model, self.max_tag_line_ch_ov, p_list, False)

        i_s = find_tag_line(CP.coord_sub,     g_input.decoupling,'y')

        i_c = find_tag_line(CP.coord_channel, g_input.decoupling,'y')

        p_list = [i_s,i_c]
        self.max_tag_line, self.tag_base_ch, self.lines_base_ch, mesh_model      = _create_lines(mesh_model,self.max_tag_line_ov,p_list,False)
        if g_input.cr !=0: 

            i_c = find_tag_line(CP.coord_channel, g_input.cr,'y')

            p_list = [i_c, CP.tag_right_c_cr[0]]
            self.max_tag_line_crust, self.tag_L_cr, self.lines_cr, mesh_model     = _create_lines(mesh_model,self.max_tag_line,p_list,False)

            if g_input.lc !=0:

                i_c = find_tag_line(CP.coord_channel,(1-g_input.lc)*g_input.cr,'y')

                p_list = [i_c, CP.tag_right_c_lcr[0]]
                self.max_tag_line_Lcrust, self.tag_L_Lcr, self.lines_lcr, mesh_model = _create_lines(mesh_model, self.max_tag_line_crust,p_list,False)

        self.line_global = np.hstack([self.lines_T, self.lines_R, self.lines_B, self.lines_L, self.lines_S, self.lines_ch, self.lines_oc, self.lines_ch_ov, self.lines_L_ov, self.lines_base_ch, self.lines_cr, self.lines_lcr])
        
        return mesh_model
        
        



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

def _create_lines(mesh_model,previous, tag_p,flag=False):
    
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