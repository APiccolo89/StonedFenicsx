import numpy as np
import time as timing
from scipy.sparse import lil_matrix
import scipy.sparse as sps
import scipy.sparse.linalg.dsolve as linsolve
from scipy.special import erf
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpmath import mpf, mpc, mp
from enum import Enum
import solver_function.shape_function as sf
from numba import njit,prange

mp.dfs = 60
"""
These class are more of a place holder -> 
[class]-[element_type]-[shape function] & [quadrature points]
--- > it can be done with only one class by the end, but, it is for the future 


WARNING: During my debugging I discovered that shape function can be affected by roundoff errors of python: 

for example: These two expressions are equivalent, but gives two different results 

C2b = -3.0*r**2-6.0*r*s+3.0*r
C2a = 3.0*r-3.0*r**2-6.0*r*s
abs((C2a-C2b)/C2a) /sim 8 % of error. 
=> The solution is to use an additional package, that handle the longfloat as if it were a C++/C software
The alternative is using a C++ code, and create a python wrapper. 
=============================================
Quick hack: 
from mpmath import mpf, mpc, mp 
mp.dps = 40 
C2b = -mpf(3.0)*mpf(r)**2-mpf(6.0)*mpf(r)*mpf(s)+mpf(3.0)*mpf(r)
C2a = mpf(3.0)*r-mpf(3.0)*r**2-mpf(6.0)*r*mpf(s)
---> end up with 0.0 as relative error. 
Food for thoughts: -> C++/C function and python wrapper? 
                   -> Using this abomination and converting the data? 
                   -> How much of these mistake are around in the literature and how much do they matter? 
==============================================
P_2^+            P_-1             P_2 

"""
class element_type(Enum):
    t_P_2_7 = 1  
    t_P_2_6 = 2 
    t_P_1_3 = 3

class shape_type(Enum):
    NNx = 1
    dNNxdr = 2
    dNNxds = 3

class field_fem():
    def __init__(self,element:int):
        self.element = element
        self.nn = self.number_of_nodes()
        self.NN = self.shape_function(element,shape_type.NNx)
        self.NNdr = self.shape_function(element,shape_type.dNNxdr)
        self.NNds = self.shape_function(element,shape_type.dNNxds)

    def number_of_nodes(self):
        if self.element == element_type.t_P_2_7.value:
            n = 7
        elif self.element == element_type.t_P_2_6.value:
            n = 6
        elif self.element == element_type.t_P_1_3.value:
            n = 3
        else:
            raise ValueError("Element not implemented.")
        return n 

    def shape_function(self,element:enumerate,shape:enumerate):
        """
        Function that compute the shape function for the element type and the shape type. 
        """
        if element == element_type.t_P_2_7.value:
            if shape.value == shape_type.NNx.value:
                return sf.NN_t_P_2_7
            elif shape.value == shape_type.dNNxdr.value:
                return sf.dNN_dr_t_P_2_7
            elif shape.value == shape_type.dNNxds.value:
                return sf.dNN_ds_t_P_2_7
            else:
                raise ValueError("Shape function not implemented for this element type.")

        if element == element_type.t_P_2_6.value:
            if shape.value == shape_type.NNx.value:
                return sf.NN_t_P_2_6
            elif shape.value == shape_type.dNNxdr.value:
                return sf.dNN_dr_t_P_2_6
            elif shape.value == shape_type.dNNxds.value:
                return sf.dNN_ds_t_P_2_6
            else:
                raise ValueError("Shape function not implemented for this element type.")

        if element == element_type.t_P_1_3.value:
            if shape.value == shape_type.NNx.value:
                return sf.NN_t_P_1_3
            else:
                return 0 

