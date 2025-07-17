import numpy as np
import matplotlib.pyplot as plt


def heat_capacity(option_C_p,T,p,it):
    if (option_C_p == 0) or (it==0): 
        # constant vriables 
        C_p = 1250.0
    elif option_C_p > 0 and option_C_p < 7:
        
        molecular_mass_fo = 140.691/1000
        molecular_mass_fa = 203.771/1000

        if option_C_p > 0 and option_C_p < 4:
            # Berman 1988 
            # forsterite 
            k_0_fo = 238.64    
            k_1_fo = -20.013e2 
            k_3_fo = -11.624e7 
        
            # fayalite 
            k_0_fa = 248.93    
            k_1_fa = -19.239e2 
            k_3_fa = -13.910e7 
        elif option_C_p > 3 and option_C_p < 7:
            # Berman & Aranovich 1996 
            # forsterite 
            k_0_fo = 233.18
            k_1_fo = -18.016e2
            k_3_fo = -26.794e7
        
            # fayalite 
            k_0_fa = 252.
            k_1_fa = -20.137e2
            k_3_fa = -6.219e7
    
        if option_C_p == 1 or option_C_p == 4: 
            # forsterite 
            molar_fraction_fa = 0.
        if option_C_p == 2 or option_C_p == 5: 
            # fayalite 
            molar_fraction_fa = 1.
        if option_C_p == 3 or option_C_p == 6: 
            # molar fraction of fayalite is 0.11 
            molar_fraction_fa = 0.11 
            
        # calculate C_p 
        C_p_fo = (k_0_fo + k_1_fo * (T**(-0.5)) + k_3_fo * (T**(-3.))) #* (1./molecular_mass_fo)
        C_p_fa = (k_0_fa + k_1_fa * (T**(-0.5)) + k_3_fa * (T**(-3.))) #* (1./molecular_mass_fa)
     
        C_p       = (1 - molar_fraction_fa) * C_p_fo + molar_fraction_fa * C_p_fa
        
    return C_p


T = np.linspace(300, 2000, 1000)
C_p_1 = heat_capacity(1, T, 0, 1)
print("C_p for option 1:", C_p_1)