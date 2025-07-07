from dataclasses import dataclass,field 
import numpy as np
import os
from numba.experimental import jitclass
from numba import int64, float64,int32, types
from typing import Tuple, List
from typing import Optional
from numba import njit, prange

data_scal = [('L',float64),
             ('v',float64),
             ('rho',float64),
             ('M',float64),
             ('T',float64),
             ('Temp',float64),
             ('stress',float64),
             ('Watt',float64),
             ('Force',float64),
             ('Cp',float64),
             ('k',float64),
             ('eta',float64),
             ('cm2myr',float64),
             ('strain',float64),
             ('ac',float64),
             ('Energy',float64)]

@jitclass(data_scal)
class Scal: 
    def __init__(self,L=1.0,
                 stress = 1e9,
                 eta = 1e22,
                 Temp  = 1.0,
                 cm2myr = 1e-2/365.25/24/60/60):
        self.cm2myr    = cm2myr
        self.L         = L 
        self.Temp      = Temp
        self.eta       = eta
        self.stress    = stress
        self.T         = self.eta/self.stress 
        self.M         = (self.stress*self.L**2)*self.T**2/self.L
        self.ac        = self.L/self.T**2
        self.rho       = self.M/self.L**3
        self.Force     = self.M*self.ac
        self.Energy    = self.Force*self.L
        self.Watt      = self.Energy/self.T
        self.strain    = 1/self.T
        self.k         = self.Watt/(self.L*self.Temp)
        self.Cp        = self.Energy/(self.Temp*self.M)

