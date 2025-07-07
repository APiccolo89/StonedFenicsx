import numpy as np
import matplotlib.pyplot as plt
import gmsh as gm
from pylab import figure, axes, pie, title, show
import os

def writing_geo(sname
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
                Channel_loop2,
                path_test
                ):
    """
    Function that write the .geo file
    Input: 
    fname: filename, string
    p      : point list 
    ll     : list of line
    sl     : line of slab
    chl    : line of channel
    lit    : line of lithosphere
    loopl  : line loop of left side model
    loopch : line loop of channel 
    loopm  : line loop of the mantle right side of slab
    loopov : line loop of overriding lithosphere
    Output: 
    .geo file in a specific location
    """

    fname = '%s.geo'%sname
    fname = os.path.join(path_test,fname)
    id_file  = open(fname,'w+')
    write_points(id_file,point_list,pn_slab,pn_an,pn_ch)

    write_lines(id_file,lines_list,subduction_lines,channel_line,lithospheric_line,boundary_lines)

    write_physical_line(id_file,Top_boundary,Right_Boundary,Bottom_Boundary,Left_Boundary,Slab_Surface,Channel_Surface,Overriding_plate_v0,Overriding_plate_v1)

    write_loopcurve_line(id_file,Left_side_loop,Channel_loop,Channel_loop2,Mantle_above,Lithosphere_loop)


    id_file.write(' \n')
    id_file.write('// Make plane surface \n')
    counter = 0
    for i in range(10,60,10):
        counter = counter + 1
        id_file.write('Plane Surface(%d) = {%d}; \n' %(counter*1000000,i))
    
    id_file.write(' \n')

    # make physical surface
    nr_surfaces = 5
    id_file.write('// Make physical surface \n')
    for i in range(1,nr_surfaces+1):
        if i == 1:
            id_file.write('Physical Surface(0) = {%d' %(i*1000000))
        elif i == nr_surfaces:
            id_file.write(',%d};' %(i*1000000))
        else:
            id_file.write(',%d' %(i*1000000))

    id_file.write('\n')
    id_file.write('\n')

    # add this line to ensure that the subduction interface is only one element wide 
    id_file.write('MeshAlgorithm Surface{2000000} = 3;')
    id_file.write('\n')
    id_file.write('MeshAlgorithm Surface{5000000} = 3;')
    id_file.write('\n')
    #id_file.write('MeshAlgorithm Surface{3000000} = 1;')


    id_file.close()


def write_points(id_file,p,pn_slab,pn_an,pn_ch):
    """
    
    
    """
    for i in range(len(p)):
        if i == 0: 
            id_file.write('// Subduction interface points \n')
        if i == pn_slab:
            id_file.write('\n')
            id_file.write('// Anchor  points \n')
        if i == pn_an:
            id_file.write('\n')
            id_file.write('// Channel  points \n')
        if i == pn_ch:
            id_file.write('\n')
            id_file.write('// Extra  Node \n')
        a = p[i][:]
        string_point = 'Point(%d) = {%s,%s,%s,%s};\n'%(a[4],a[0],a[1],a[2],a[3])
        id_file.write(string_point)





def write_lines(id_file,ll,sl,cl,litl,bl):
    """
    
    
    """
    id_file.write('\n')
    id_file.write('\n')
    id_file.write('// Lines \n')

    for i in range(len(ll)): 
        a = ll[i][:]
        if a[2] == bl[0]:
            id_file.write('// Boundary Line \n')
            id_file.write('\n')
        if a[2] == sl[0]: 
            id_file.write('\n')
            id_file.write('// Slab Line \n')
            id_file.write('\n')
        if a[2] == cl[0]: 
            id_file.write('\n')
            id_file.write('// Channel Line \n')
            id_file.write('\n')
        if a[2] == litl[0]: 
            id_file.write('\n')
            id_file.write('// Lit Line \n')
            id_file.write('\n')


        string_line = 'Line(%d) = {%d,%d};\n'%(a[2],a[0],a[1])
        id_file.write(string_line)

    

def write_physical_line(id_file,Tb,Rb,Bb,Lb,SS,CS,OPv0,OPv1):
    """
        
        
        
    """
    def closure_string(L):
        # I love closure, but, for whoever tried to track my crime against humanity, i am pretty aware that the variable that are defined within
        # the scope of the main function act as if they were global. This is a message for the future generation => remember to not use variable defined
        # in the scope of the big function. 

        string_X = 'Physical Line(%d) = {'%(L[0][0])
        if len(L[1])==1:
            string_X = "%s%s"%(string_X,'%d};\n'%L[1][0])

        else:
            for i in range(len(L[1])):
                if i == len(L[1])-1:
                    string_X = "%s%s"%(string_X,'%d};\n'%L[1][i])
                else:                
                    string_X = "%s%s"%(string_X,'%d,'%L[1][i])

        
        L = []
        return string_X 
    # One day, I will generate classes {Point,Line,Physical Line,Loop,Physical Surface} => For generalising a script, seems easy. 
    Top_string = closure_string(Tb)
    Right_string = closure_string(Rb)
    Bottom_string = closure_string(Bb)
    Left_string  = closure_string(Lb)
    Subduction_string = closure_string(SS)
    Channel_string = closure_string(CS)
    OVPv0_string = closure_string(OPv0)
    OVPv1_string = closure_string(OPv1)

    # Top boundary
    id_file.write('\n')
    id_file.write('\n')
    id_file.write('//Physical Lines \n')
    id_file.write('\n')
    id_file.write('//Boundary Lines \n')
    id_file.write('\n')
    id_file.write(Top_string)
    id_file.write(Right_string)
    id_file.write(Bottom_string)
    id_file.write(Left_string)
    id_file.write('\n')
    id_file.write('//Subduction Lines \n')
    id_file.write(Subduction_string)
    id_file.write('\n')
    id_file.write('//Channel Lines \n')
    id_file.write(Channel_string)
    id_file.write('\n')
    id_file.write('//OVP Lines \n')
    id_file.write(OVPv0_string)
    id_file.write(OVPv1_string)
    id_file.write('\n')

def  write_loopcurve_line(id_file,LsL,CL,CL2,MaL,LL):
    def closure_string(L):
        # I love closure, but, for whoever tried to track my crime against humanity, i am pretty aware that the variable that are defined within
        # the scope of the main function act as if they were global. This is a message for the future generation => remember to not use variable defined
        # in the scope of the big function. 

        string_X = 'Line Loop(%d) = {'%(L[0][0])
        if len(L[1])==1:
            string_X = "%s%s"%(string_X,'%d};\n'%L[1][0])

        else:
            for i in range(len(L[1])):
                if i == len(L[1])-1:
                    string_X = "%s%s"%(string_X,'%d};\n'%L[1][i])
                else:                
                    string_X = "%s%s"%(string_X,'%d,'%L[1][i])

        
        L = []
        return string_X 


    string_LsL = closure_string(LsL)
    string_CL  = closure_string(CL)
    string_MaL = closure_string(MaL)
    string_LL  = closure_string(LL)
    string_CL2  = closure_string(CL2)


    id_file.write('\n')
    id_file.write('\n')
    id_file.write('//Loop Lines \n')
    id_file.write('\n')
    id_file.write(string_LsL)
    id_file.write(string_CL)
    id_file.write(string_MaL)
    id_file.write(string_LL)
    id_file.write(string_CL2)
    id_file.write('\n')






