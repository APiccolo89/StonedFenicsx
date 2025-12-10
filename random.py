


import numpy as np 
import matplotlib.pyplot as plt 

def compute_heat_capacity(C0,C1,C2,C3,C4,C5,T):
    
    Cp = C0 + C1 * T**(-1/2) + C2 * T**(-2) + C3 * T**(-3) + C4 * T + C5 * T**2
    
    return Cp 

T = np.linspace(273.15,1873.15,num=1000)

C0_Ol = 1.6108 * 1e3 
C1_Ol = -1.24788 * 1e4 
C2_Ol = 0.0 
C3_Ol = -1.728477*1e9 
C4_Ol = 0.0 
C5_Ol = 0.0 
# ---
# Clino ( Augite) 
# --- 
C0_au = 2.1715   * 1e3 
C1_au = -2.22716 * 1e4
C2_au = 1.1333   * 1e6 
C3_au = 0.0 
C4_au = -4.555 * 1e-1 
C5_au = 1.299 * 1e-4
# ---
# Plagioclase 
C0_pg = 1.85757  * 1e3 
C1_pg = -1.64946 * 1e4
C2_pg = -5.061   * 1e6 
C3_pg = 0.0 
C4_pg = -3.324 * 1e-1 
C5_pg = 1.505 * 1e-4
fr_ol = 0.15; fr_au = 0.2; fr_pg = 0.65

C0 = fr_ol * C0_Ol + fr_au * C0_au + fr_pg * C0_pg
C1 = fr_ol * C1_Ol + fr_au * C1_au + fr_pg * C1_pg
C2 = fr_ol * C2_Ol + fr_au * C2_au + fr_pg * C2_pg
C3 = fr_ol * C3_Ol + fr_au * C3_au + fr_pg * C3_pg
C4 = fr_ol * C4_Ol + fr_au * C4_au + fr_pg * C4_pg
C5 = fr_ol * C5_Ol + fr_au * C5_au + fr_pg * C5_pg

Cp_M  = compute_heat_capacity(C0,C1,C2,C3,C4,C5,T)
Cp_O  = compute_heat_capacity(C0_Ol,C1_Ol,C2_Ol,C3_Ol,C4_Ol,C5_Ol,T)
Cp_au = compute_heat_capacity(C0_au,C1_au,C2_au,C3_au,C4_au,C5_au,T)
Cp_pg = compute_heat_capacity(C0_pg,C1_pg,C2_pg,C3_pg,C4_pg,C5_pg,T)
Cp_2 = fr_ol * Cp_O + fr_au * Cp_au + fr_pg * Cp_pg 