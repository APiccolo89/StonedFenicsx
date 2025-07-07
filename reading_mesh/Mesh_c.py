from typing import Union
import gmsh as gm
import numpy as np
import matplotlib.pyplot as plt 
import input
from solver_function.function_fem import field_fem
from solver_function.function_fem import element_type
from dataclasses import dataclass, field
from typing import List
import numba 
from numba import njit, prange
from numba.experimental import jitclass
from numba import float64, int32, int64


mesh_spec = [
    ('node_num', int32[:]),
    ('x', float64[:]),
    ('y', float64[:]),
    ('el', int32[:]),
    ('el_con', int32[:, :]),
    ('nel', int32),
    ('nv', int32),
    ('xP', float64[:]),
    ('yP', float64[:]),
    ('el_conP', int32[:, :]),
    ('Phase', int32[:]),
    ('Edge_element', int32[:,:]),
]

@jitclass(mesh_spec)
class MESH:
    def __init__(self, node_num, x, y, el, el_con, nel, nv):
        self.node_num = node_num
        self.x = x
        self.y = y
        self.el = el
        self.el_con = el_con
        self.nel = nel
        self.nv = nv
        # Initialize with dummy arrays (to be set later)
        self.xP = np.zeros(1, dtype=np.float64)
        self.yP = np.zeros(1, dtype=np.float64)
        self.el_conP = np.zeros((1, 1), dtype=np.int32)
        self.Phase = np.zeros(nv, dtype=np.int32)
        self.Edge_element = np.zeros((nel,4), dtype=np.int32)



spec = [
    ('ndim', int32),
    ('mV', int32),
    ('mP', int32),
    ('mT', int32),
    ('vel_fem', int32),
    ('temp_fem', int32),
    ('pres_fem', int32),
    ('ndofV', int32),
    ('ndofP', int32),
    ('ndofT', int32),
    ('NfemV', int32),
    ('NfemP', int32),
    ('NfemT', int32),
    ('qcoords_r', float64[:]),
    ('qcoords_s', float64[:]),
    ('qweights', float64[:]),
    ('Vnodes', float64[:,:]),
    ('nq', int32),
    ('nqel', int32),
]

@jitclass(spec)
class Computational_Data:
    def __init__(self, ndim, mV, mP, mT, elementV, elementT, elementP,nel):
        self.ndim = ndim
        self.vel_fem = elementV
        self.temp_fem = elementT
        self.pres_fem = elementP
        self.mV = mV
        self.mP = mP
        self.mT = mT
        self.ndofV = ndim
        self.ndofP = 1
        self.ndofT = 1
        self.NfemV = 0
        self.NfemP = 0
        self.NfemT = 0

        qcoords_r, qcoords_s, qweights, nqel = self.find_quadrature_points(6)
        self.qcoords_r = np.array(qcoords_r, dtype=np.float64)
        self.qcoords_s = np.array(qcoords_s, dtype=np.float64)
        self.qweights = np.array(qweights, dtype=np.float64)
        self.nq = nqel
        self.nqel = nel*nqel
        self.Vnodes = np.zeros((self.mV, 2), dtype=np.float64)
        self.Vnodes[:, 0] = np.array([0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0/3.], dtype=np.float64)
        self.Vnodes[:, 1] = np.array([0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 1.0/3.], dtype=np.float64)

    def _update_(self, nv, nel):
        self.NfemV = nv * self.ndofV
        self.NfemP = 3 * nel * self.ndofP
        self.NfemT = (nv - nel) * self.ndofT
        return self

    def find_quadrature_points(self, nq):
        nb1 = 0.816847572980459
        nb2 = 0.091576213509771
        nb3 = 0.108103018168070
        nb4 = 0.445948490915965
        nb5 = 0.109951743655322 / 2.
        nb6 = 0.223381589678011 / 2.

        qcoords_r = [nb1, nb2, nb2, nb4, nb3, nb4]
        qcoords_s = [nb2, nb1, nb2, nb3, nb4, nb4]
        qweights = [nb5, nb5, nb5, nb6, nb6, nb6]
        return qcoords_r, qcoords_s, qweights, nq

def compute_pressure_field_mesh(M:MESH,CD:Computational_Data):

    iconP = np.zeros((M.nel,CD.mP),dtype=np.int32)
    
    xP = np.zeros(CD.NfemP,dtype=np.float64)     # x coordinates
    
    yP = np.zeros(CD.NfemP,dtype=np.float64)     # y coordinates

    counter=1

    for iel in range(0,M.nel):
        
        xP[counter-1] = M.x[M.el_con[iel,0]-1]
        
        yP[counter-1] = M.y[M.el_con[iel,0]-1]
        
        iconP[iel,0] = counter
        
        counter += 1
        
        xP[counter-1] = M.x[M.el_con[iel,1]-1]
        
        yP[counter-1] = M.y[M.el_con[iel,1]-1]
        
        iconP[iel,1] = counter
        
        counter += 1
        
        xP[counter-1] = M.x[M.el_con[iel,2]-1]
        
        yP[counter-1] = M.y[M.el_con[iel,2]-1]
        
        iconP[iel,2] = counter
        
        counter+=1
    
    M.xP = xP
    
    M.yP = yP
    
    M.el_conP = iconP
    

    return M
# Crouzeix-Raviart elements for Stokes, P_2 for Temp.
# the internal numbering of the nodes is as follows:
#
#  P_2^+            P_-1             P_2 
#
#  02           #   02           #   02
#  ||\\         #   ||\\         #   ||\\
#  || \\        #   || \\        #   || \\
#  ||  \\       #   ||  \\       #   ||  \\
#  04   03      #   ||   \\      #   04   03
#  || 06 \\     #   ||    \\     #   ||    \\
#  ||     \\    #   ||     \\    #   ||     \\
#  00==05==01   #   00======01   #   00==05==01


# Shape function for the velocity field: second order triangle element with barycentral node

def _find_boundary_edge(M:MESH,CD:Computational_Data,side:str):
    """
    Find the left boundary edge of the mesh.
    """
    # Initialize an empty list to store the left boundary edges
    boundary_edges = []
    if side == 'left':
        ck = np.min(M.x)
        flag = 1
    elif side == 'right':
        ck = np.max(M.x)
        flag = 2
    elif side == 'top': 
        ck = np.max(M.y)
        flag = 3
    elif side == 'bottom':  
        ck = np.min(M.y)
        flag = 4


    # Loop through each element in the mesh
    for iel in range(M.nel):
        # Get the coordinates of the vertices of the element
        x1, y1 = M.x[M.el_con[iel, 0] - 1], M.y[M.el_con[iel, 0] - 1]
        x2, y2 = M.x[M.el_con[iel, 1] - 1], M.y[M.el_con[iel, 1] - 1]
        x3, y3 = M.x[M.el_con[iel, 2] - 1], M.y[M.el_con[iel, 2] - 1]
        c = 0 
        if side == 'left' or side=='right':
            a1 = x1
            a2 = x2
            a3 = x3
        elif side == 'top' or side=='bottom':
            a1 = y1
            a2 = y2
            a3 = y3 

        vertex = np.array([0, 0, 0], dtype=np.int32)

        # Check if any of the vertices are on the left boundary (x=0)
        if (a1 == ck) or (a2 == ck) or (a3 == ck):
            if (a1 == ck): 
                c += 1 
                vertex[0] = 1
            if (a2 == ck):
                c += 1
                vertex[1] = 1
            if (a3 == ck):
                c += 1
                vertex[2] = 1
            
            if c==2:
                M.Edge_element[iel,0] = flag
                if vertex[0] == 1 and vertex[1] == 1:
                    M.Edge_element[iel,1] = 0
                    M.Edge_element[iel,2] = 5
                    M.Edge_element[iel,3] = 1
                elif vertex[0] == 1 and vertex[2] == 1:
                    M.Edge_element[iel,1] = 0
                    M.Edge_element[iel,2] = 4
                    M.Edge_element[iel,3] = 2
                elif vertex[1] == 1 and vertex[2] == 1:
                    M.Edge_element[iel,1] = 1
                    M.Edge_element[iel,2] = 5
                    M.Edge_element[iel,3] = 2
    
    return M
