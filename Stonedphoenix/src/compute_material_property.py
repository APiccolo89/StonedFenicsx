# calculate the thermal structure of the ocean according to Richards et al., 2018 JGR: Solid Earth 
# we use a time- and space-centered Crank-Nicholson finite-difference scheme with a predictor-corrector step (Press et al., 1992)

# import all the constants and defined model setup parameters 

# modules
import numpy as np
import matplotlib.pyplot as plt
import time as timing
import scipy.linalg as la 
import scipy.sparse as sps
import scipy.sparse.linalg.dsolve as linsolve
from numba import njit
from numba import jit, prange
from scipy.optimize import bisect
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
@njit
def heat_conductivity(pdb,T,p,ph):    
    
    if pdb.option_k[ph] == 0:
        # constant variables 
        
        k =pdb.k0[ph]
    
    elif pdb.option_k[ph] >1: 
        
        k = (pdb.k0[ph] + pdb.a[ph] * p)
        
        if pdb.option_k[ph] == 3:
           
           # radiative component of the conductivity
            k_h_radiative = 0.
            
            for i in range(0,4):
            
                k_h_radiative = k_h_radiative + pdb.d[ph,i] * (T)**i
     
            k = k + k_h_radiative
             
        
    return k 
#---------------------------------------------------------------------------------

@njit
def heat_capacity(pdb,T,p,ph):
    if (option_C_p == 0) 
        # constant vriables 
        C_p = pdb.C_p[ph]
    else:
        C_p = pdb.C0[ph] + pdb.C1[ph] * (T**(-0.5)) + pdb.C3[ph] * (T**(-3.))

        
    return C_p
#---------------------------------------------------------------------------------
@njit
def density(pdb,T,p,ph):
    rho_0 = pdb.rho0[ph] 
    
    if pdb.option_rho[ph] == 0
        # constant variables 
        return rho_0 
    else :
        # calculate rho
        rho     = rho_0 * np.exp( - ( pdb.alpha0[ph] * (T - pdb.Tref) + (pdb.alpha2[ph]/2.) * ( T**2 - pdb.Tref**2 )))
        if pdb.option_rho[ph] == 2:
            # calculate the pressure dependence of the density
            Kb = (2*100e9*(1+0.25))/(3*(1-0.25*2))
            rho = rho * np.exp(p/Kb)    
    
    return rho

#----------------------------------------------------------------------------------

@njit
def density(option_rho,T,p,it):
    if (option_rho == 0) or (it==0):
        # constant variables 
        rho = 3300. + (T - T)
    elif option_rho == 1 or option_rho == 2:
        # constants 
        rho_0   = 3330
        alpha_0 = 2.832e-5
        alpha_1 = 3.79e-8 
        # calculate rho
        rho     = rho_0 * np.exp( - ( alpha_0 * (T - 273.15) + (alpha_1/2.) * ( T**2 - 273.15**2 ) ) )
        if option_rho == 2:
            # calculate the pressure dependence of the density
            Kb = (2*100e9*(1+0.25))/(3*(1-0.25*2))
            rho = rho * np.exp(p/Kb)    
    return rho


@njit
def compute_viscosity(e,T,P,B,n,E,V,R):

    t = -1+1/n

    eta = 0.5*B**(-1/n)*e**t*np.exp((E+P*V)/(n*R*T))

    return eta 
#---------------------------------------------------------------------------------
# TILL STOKES BETTER TO NOT TOUCH THIS FUNCTION
@njit
def viscosity(exx:float,eyy:float,exy:float,T:float,P:float,ctrl,p_data,it:int,ph:int):#,imat):
    ck = 0
    e=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)
    
    
    
    rheo_type = ctrl.rheology
    eta_max   = ctrl.eta_max
    eta_min   = ctrl.eta_min
    Bdif = p_data.Bdif[ph]
    Edif = p_data.Edif[ph]
    Vdif = p_data.Vdif[ph]

    R    = ctrl.R
    Bdis = p_data.Bdis[ph]
    Edis = p_data.Edis[ph]
    Vdis = p_data.Vdis[ph]
    n    = p_data.n[ph]

    # Certain area of the model does not have a real velocity field
    if e == 0: 
        e = 1e-21
    
    if (it == 0) or (rheo_type== 0):
        return ctrl.eta_def

    if p_data.eta[ph] != -1e23:
        return p_data.eta[ph]

          
    if rheo_type== 1:
        # diffusion creep 
        eta_dif = compute_viscosity(e,T,P,Bdif,1,Edif,Vdif,R)

        val=1/(1/eta_max+1/eta_dif)
        
    if rheo_type == 3: 
        # dislocation creep 
        eta_dis = compute_viscosity(e,T,P,Bdis,n,Edis,Vdis,R)
        val=1/(1/eta_max+1/eta_dis)
        
    if rheo_type == 2:
        # dislocation & diffusion creep 
        eta_dif = compute_viscosity(e,T,P,Bdif,1,Edif,Vdif,R)
        eta_dis = compute_viscosity(e,T,P,Bdis,n,Edis,Vdis,R)
        eta_eff = (1/eta_dif+1/eta_dis)**(-1)
        val=1/(1/eta_max+1/eta_eff)
    
    if rheo_type == 4: 

        val,tau = point_iteration(eta_max,eta_min,e,T,P,Bdif,Edif,Vdif,Bdis,n,Edis,Vdis,R)
          
    return val
#---------------------------------------------------------------------------------

@njit
def _find_tau_guess(cdf,cds,n,eta_max,e):
    """
    input:
    n -> stress exponent
    Bdisl -> dislocation creep parameter
    Qdisl -> dislocation activation energy
    Bdif  -> diffusion creep parameter
    Qdif  -> diffusion activation energy
    eta_max -> maximum viscosity
    e -> strain rate
    T -> temperature
    output:
    t -> tau_guess
    => First compute the minimum and maximum viscosity for the given 
    input strain rate and temperature.
    -> Then compute the relative stresses associated with these mecchanism 
    -> compute the average value of the stresses
    ---> The real stress is between the value defined by the armonic average viscosity and the 
    value defined by the minimum viscosity between the involved mechanism 
    """
    # Compute the viscosity associated with the dislocation and diffusion creep
    # mechanism
    etads=0.5*cds**(-1/n)*e**((1-n)/n)
    etadf=0.5*cdf**(-1)
    # Compute the viscosity associated with the maximum viscosity
    # mechanism
    av_vis = (1/etadf+1/etads+1/eta_max)**(-1)
    
    if etadf<etads:
        min_vis = etadf
    else:
        min_vis = etads

    if min_vis>eta_max:
        min_vis = eta_max


    tau_min = 2*av_vis*e
    tau_max = 2*min_vis*e 

    return tau_min,tau_max


#---------------------------------------------------------------------------------

# Define the function f(tau) for root-finding
@njit
def f(tau, compliance_disl, compliance_diff, B_max, e, n):
    """Equation to solve for tau."""
    return (e - (compliance_disl * tau**n + compliance_diff * tau + B_max * tau)) / e

# Custom bisection method
@njit
def bisection_method(a, b, tol, max_iter, compliance_disl, compliance_diff, B_max, e, n):
    """Perform bisection method to find the root of f in the interval [a, b]."""
    fa = f(a, compliance_disl, compliance_diff, B_max, e, n)
    fb = f(b, compliance_disl, compliance_diff, B_max, e, n)
    if (np.abs(fa)<tol) or (np.abs(fb)<tol):
        return a if np.abs(fa)<tol else b
    if fa * fb > 0:
        raise ValueError("Function has the same sign at the endpoints a and b")

    for _ in range(max_iter):
        # Compute midpoint
        c = (a + b) / 2
        fc = f(c, compliance_disl, compliance_diff, B_max, e, n)

        # Check for convergence
        if abs(fc) < tol:
            return c
        elif fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return (a + b) / 2  # Return midpoint after max_iter iterations
#---------------------------------------------------------------------------------

# Main point_iteration function
@njit
def point_iteration(eta_max: float,
                    eta_min: float,
                    e: float,
                    T: float,
                    P: float,
                    Bdif: float,
                    Edif: float,
                    Vdif: float,
                    Bdis: float,
                    n: float,
                    Edis: float,
                    Vdis: float,
                    R: float) -> float:
    """
    Optimized point-wise iteration to compute viscosity, real stress.
    """

    # Precompute constant values to avoid repeated calculations
    B_max = 1 / (2 * eta_max)
    compliance_disl = Bdis * np.exp(-(Edis+P*Vdis) / (R * T))
    compliance_diff = Bdif * np.exp(-(Edif+P*Vdif) / (R * T))

    # Find tau_guess using custom bisection method
    tau_min, tau_max = _find_tau_guess(compliance_diff, compliance_disl, n, eta_max, e)
    if abs(tau_max - tau_min) / (tau_max + tau_min) < 1e-6:
        tau = (tau_min + tau_max) / 2
    else:
        tol = 1e-8
        max_iter = 100
        tau = bisection_method(tau_min, tau_max, tol, max_iter, compliance_disl, compliance_diff, B_max, e, n)

    # Compute the viscosity and real stress
    e_diff = compliance_diff * tau
    e_dis = compliance_disl * tau ** n
    e_max = B_max * tau

    tau_total = tau + 2 * eta_min * e
    eta = tau_total / (2 * e)

    return eta, tau_total
#---------------------------------------------------------------------------------
def _check_iteration(ctrl,ph,e,T,P,it):

    
    rheo_type = ctrl.rheology
    eta_max   = ctrl.eta_max
    eta_min   = ctrl.eta_min
    Bdif = ph.Bdif[0]
    Edif = ph.Edif[0]
    Vdif = ph.Vdif[0]

    R    = ctrl.R
    Bdis = ph.Bdis[0]
    Edis = ph.Edis[0]
    Vdis = ph.Vdis[0]
    n    = ph.n[0]
    eta_dif = compute_viscosity(e,T,P,Bdif,1,Edif,Vdif,R)
    eta_dis = compute_viscosity(e,T,P,Bdis,n,Edis,Vdis,R)
    eta_effA = (1/eta_dif+1/eta_dis+1/eta_max)**(-1)
    eta_effB = np.min([eta_dif, eta_dis, eta_max])
    
    
    B_max = 1 / (2 * eta_max)
    compliance_disl = Bdis * np.exp(-(Edis+P*Vdis) / (R * T))
    compliance_diff = Bdif * np.exp(-(Edif+P*Vdif) / (R * T))
    # cross check with compliance
    eta_dif2 = 0.5*compliance_diff**(-1)
    eta_dis2 = 0.5*compliance_disl**(-1/n)* np.sqrt(e)**((1-n)/n)
    tau_min, tau_max = _find_tau_guess(compliance_diff, compliance_disl, n, eta_max, e)
    tau_max2 = 2*eta_effB*e
    tau_min2 = 2*eta_effA*e
    tau = bisection_method(tau_min, tau_max, 1e-6, 100, compliance_disl, compliance_diff, B_max, e, n)
    eta  = tau / (2 * e)
    tau_min2 = 2*eta_min*e
    eta_2 = (tau + 2 * eta_min * e) / (2 * e)

    e_diff = compliance_diff * tau
    e_dis = compliance_disl * tau ** n
    e_max = B_max * tau
    r = e/e-(e_diff + e_dis + e_max)/e 
    return eta_2






if __name__ == '__main__':
    import sys
    import os

    # Append absolute path to the folder (NOT to the file)
    sys.path.append(os.path.abspath("material_property"))
    sys.path.append(os.path.abspath("solver_function"))

    # Now import the module by filename (without .py)
    import phase_db as pdb
    import numerical_control as nc
    ctrl = nc.NumericalControls(rheology=3) 

    ph = pdb.PhaseDataBase(2)
    pdb._generate_phase(ph,2,id=0,name_diffusion='Hirth_Dry_Olivine_diff',name_dislocation='Hirth_Dry_Olivine_disl')
    pdb._generate_phase(ph,2,id=1,name_diffusion='Van_Keken_diff',name_dislocation='Hirth_Dry_Olivine_disl')
    exx,eyy,exy = 1e-15,1e-15,1e-15
    e=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)

    T = np.linspace(273.15, 1300+273.15, 200)  # Temperature range from 0 to 1600 K
    P = np.linspace(0, 20e9, 1000)  # Pressure range from 0 to 1000 MPa
    
    eta_iter = np.zeros([len(T),len(P)], dtype=np.float64)
    for it in range(len(T)):
        for it2  in range(len(P)):
            # Compute the minimum and maximum viscosity for the current temperature and pressure
            eta_iter[it,it2] = _check_iteration(ctrl,ph,e,T[it],P[it2],it)

    T = T - 273.15  # Convert to Celsius for plotting
    plt.contourf(T, P/1e6, np.log10(np.transpose(eta_iter)), levels=30, cmap='viridis')
    plt.colorbar(label='Viscosity (Pa.s)')
    plt.xlim(0, 1300)  # Limit temperature to 1300 K for better visibility
    plt.xlabel('Temperature (deg C)')
    plt.xlabel('Pressure (MPa)')

    plt.title('Viscosity vs Temperature')
    plt.legend()
    print("Viscosity at T=1300K:", eta_iter[np.where(T==1300)[0][0]])

