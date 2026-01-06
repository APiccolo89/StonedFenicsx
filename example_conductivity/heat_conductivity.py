import numpy as np 
import matplotlib.pyplot as plt 
'''
Small script for comparing different options for computing thermal conductivity, density and thermal expansivity
as a function of Pressure and Temperature.
Authored by Andrea Piccolo Somewhen between  2025-2026
'''
from scipy.optimize import fsolve 

def compute_diffusivity_T(pdb, T):
    """Compute thermal diffusivity based on empirical formula. Ora, sti stronzi devono smetterla di rilasciare le porco dio di
    costanti usando le loro SI di sta merda: USATE M S KG standard, perche' in ogni fottuta equazione termodinamica si usano quelle
    ma che cazzo di merda hanno nel cervello?  Non solo, le equazioni sono sbagliate, dio maiale infame, Korenaga 2016 da una versione corretta
    a quanto pare era T/c no T*c, ma chissene se devo perdere il mio tempo, tanto sono immortale, vero? Che si fottano, poi appena vedono un mio articolo
    fanno la punta al cazzo per ragioni del cazzo. The simplest bug of all the time, forgot a parenthesis.
    """
    # Data from Fosterite
    a = 0.565/1e6  
    b = 0.67/1e6
    c = 590.0
    d = 1.4/1e6
    e = 135.0
    T = T-273.15
    D = a + b * np.exp(-T/c) + d * np.exp(-T/e)
    
    return D  # in mm^2/s

def compute_radiative_conductivity_T(T,d):
    """Compute radiative conductivity based on empirical formula.
    """
    def A(d):
        return 1.8 * (1 - np.exp(-d**1.3 / 0.15)) - (1 - np.exp(-d**0.5 / 5.0))

    def B(d):
        return 11.7 * np.exp(-d / 0.159) + 6.0 * np.exp(-d**3 / 10.0)

    def T_A(d):
        return 490.0 + 1850.0 * np.exp(-d**0.315 / 0.825) + 875.0 * np.exp(-d / 0.18)

    def T_B(d):
        return 2700.0 + 9000.0 * np.exp(-d**0.5 / 0.205)

    def x_A(d):
        return 167.5 + 505.0 * np.exp(-d**0.5 / 0.85)

    def x_B(d):
        return 465.0 + 1700.0 * np.exp(-d**0.94 / 0.175) 

     # density in g/cm3 for Fosterite      

    cfA =A(d) * np.exp(-(T-T_A(d))**2/ (2*x_A(d) ** 2 ))
    cfB = B(d) * np.exp(-(T - T_B(d))**2 / (2* x_B(d)**2))
    krad = A(d) * np.exp(-(T-T_A(d))**2/ (2*x_A(d) ** 2 )) + B(d) * np.exp(-(T - T_B(d))**2 / (2* x_B(d)**2))

    return krad # in W/mK

def compute_diffusivity_P(P):
    a = 0.05/1e9 
    
    D = np.exp(a * P)

    return D   # it is the dimensionless, i hate these fuckers 


def compute_thermal_capacity_T(T):
    """_summary_
    Altro Rant: troppo difficile fare le cose con le cazzo di SI jesus christ, ma che cazzo di merda hanno nel cervello questi stronzi, poi si lamentano che gli articoli non vengono accettati
    1) Kj = 1000 J
    Args:
        T (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    
    a = 1.6108   * 1e3
    b = -12.4788 * 1e3
    c = 1728477  * 1e3
    Cp = (a + b * T ** (-0.5) + c * T ** (-3))
    
    return Cp
    

def find_root_V0(P, K0, K0p):
    
    def f(V_V0):
        return (3.0/2.0)*K0*((V_V0)**(7.0/3.0)-(V_V0)**(5.0/3.0))*(1.0 + (3.0/4.0)*(K0p -4.0)*((V_V0)**(2.0/3.0)-1.0)) - P
    
    V_V0_initial_guess = 1.0 - P/(3.0*K0)  # Initial guess based on small compression
    
    V_V0_solution = fsolve(f, V_V0_initial_guess)[0]
    
    return V_V0_solution

# Grose and Alfonso 
def compute_isothermal_volume_change(P, K0, K0p): 
    
    a = 1.0 
    
    a = find_root_V0(P, K0, K0p)
    V_V0 = a    
    return V_V0

def compute_thermal_expansivity_pressure_dependent(V_V0,Gr0):
    
    alphaP = V_V0 * np.exp((Gr0+1.0)*(V_V0**(-1) - 1.0)) 
    
    return alphaP

def compute_thermal_expansivity_pressure_dependent2(P,K0):
    Kb     = 3.63*1e-2/1e9

    alphaP = np.exp(-P * Kb)
    
    return alphaP

def compute_thermal_expansitivity_temperature_dependent(T, alpha0, alpha1 , T0 , q):
    
    alphaT = alpha0   + alpha1 * (T-T0) 

    return alphaT
    

def compute_int_thermal_expansitivity_temperature_dependent(T, alpha0, alpha1 , T0 , q):
    
    alphaT = alpha0*(T-T0)   + 0.5*alpha1 * (T**2-T0**2)

    return alphaT

#--------------------------------------------------------------------------------------# 
# Main script


T = np.linspace(298.15, 1800+273.15, 1001) #[K]

k_rad = np.zeros(len(T))

k_rad = compute_radiative_conductivity_T(T,0.5)

P = np.linspace(0, 24e9, 600) #[Pa]

K0 = 130e9 #[Pa]

K0p = 4.8  

V_V0 = np.zeros(len(P))

alpha_p = np.zeros(len(P))

alpha_p2 = np.zeros(len(P))

for i in range(len(P)):

    V_V0[i] = compute_isothermal_volume_change(P[i], K0, K0p)
    
    alpha_p2[i] = compute_thermal_expansivity_pressure_dependent2(P[i],K0)
    
    alpha_p[i] = compute_thermal_expansivity_pressure_dependent(V_V0[i], Gr0=6.0)


alpha_T = compute_thermal_expansitivity_temperature_dependent(T, alpha0=2.832e-5,alpha1 = 0.758e-8 , T0=273.15, q=1.0)

rho_0 = 3330.0 #[kg/m3]

rho    = np.zeros((len(P), len(T)))
rho2   = np.zeros((len(P), len(T)))
Kb     = (2*130e9*(1+0.25))/(3*(1-0.25*2))
alpha0 =  2.832e-5
alpha1 =  0.758e-8 #[1/K2]
k      = np.zeros((len(P), len(T)))
k_rad  = np.zeros((len(P), len(T))) 
Cp  = np.zeros((len(P), len(T))) 
kp = np.zeros((len(P), len(T)))
k2 = np.zeros((len(P),len(T)))
alpha_total = np.zeros((len(P), len(T)))
alpha_total2 = np.zeros((len(P), len(T)))


for i in range(len(P)):
    for j in range(len(T)):
        
        rho[i,j]   = rho_0 * V_V0[i] * (1-alpha_p[i] * compute_int_thermal_expansitivity_temperature_dependent(T[j], alpha0, alpha1 , 273.15 , q=1.0))
        rho2[i,j]   = rho_0 * np.exp(P[i]/Kb) * (1- alpha_p2[i] * compute_int_thermal_expansitivity_temperature_dependent(T[j], alpha0, alpha1 , 273.15 , q=1.0))
        alpha_total[i,j] =  alpha_T[j]*alpha_p[i]
        alpha_total2[i,j] =  alpha_T[j]*alpha_p2[i]
        k[i,j]     = compute_diffusivity_T(P[i], T[j]) * rho[i,j] * compute_thermal_capacity_T(T[j]) * compute_diffusivity_P(P[i])
        k_rad[i,j] = compute_radiative_conductivity_T(T[j],0.5)
        Cp[i,j]    = compute_thermal_capacity_T(T[j])
        kp[i,j]    = compute_diffusivity_P(P[i])
        k2[i,j]    =  compute_diffusivity_T(P[i], T[j]) * rho2[i,j] * compute_thermal_capacity_T(T[j]) * compute_diffusivity_P(P[i])
        


# ---- adiabatic gradient of an isothermal mantle ---- # 

z = np.linspace(0, 660e3, 1000)  # depth in meters
TA = np.ones(len(z)) * 1350 + 273.15  # temperature in K
TB = np.ones(len(z)) * 1350 + 273.15  # temperature in K
V_V0 = np.ones(len(z))
PA = rho_0 * 9.81 * z  # pressure in Pa
PB = rho_0 * 9.81 * z  # pressure in Pa
rho_A = np.ones(len(z)) * rho_0
rho_B = np.ones(len(z)) * rho_0
alpha_A = np.ones(len(z))
alpha_B = np.ones(len(z))
CpA = np.ones(len(z))
CpB = np.ones(len(z))

res = 1.0 

while res > 1e-6:
    for i in range(len(z)):
        aTA = compute_int_thermal_expansitivity_temperature_dependent(TA[i], alpha0, alpha1 , 273.15 , q=1.0)
        aTB = compute_int_thermal_expansitivity_temperature_dependent(TB[i], alpha0, alpha1 , 273.15 , q=1.0)
        
        alpha_TA = compute_thermal_expansitivity_temperature_dependent(TA[i], alpha0, alpha1 , 273.15 , q=1.0)
        alpha_TB = compute_thermal_expansitivity_temperature_dependent(TB[i], alpha0, alpha1 , 273.15 , q=1.0)
        
        aPA = compute_thermal_expansivity_pressure_dependent(V_V0[i], Gr0=6.0)
        aPB = compute_thermal_expansivity_pressure_dependent2(PB[i], K0)
        
        CpA[i] = compute_thermal_capacity_T(TA[i])
        CpB[i] = compute_thermal_capacity_T(TB[i])
        
        V_V0[i] = compute_isothermal_volume_change(PA[i], K0, K0p)
        rho_A[i] = rho_0 * V_V0[i] * (1 - aPA * aTA)
        alpha_A[i] = aPA * alpha_TA
        
        rho_B[i] = rho_0 * np.exp(PB[i]/Kb) * (1 - aPB * aTB)
        alpha_B[i] = aPB * alpha_TB
    
    T_newA = (1350 + 273.15) * np.exp(alpha_A/rho_A/CpA * PA)
    T_newB = (1350 + 273.15) * np.exp(alpha_B/rho_B/CpB * PB)
    
    P_newA = np.cumsum(rho_A * 9.81 * np.mean(np.diff(z, prepend=0)))
    P_newB = np.cumsum(rho_B * 9.81 * np.mean(np.diff(z, prepend=0)))
    
    resA = np.linalg.norm(T_newA-TA,2)/np.linalg.norm(T_newA+TA,2)
    resB = np.linalg.norm(T_newB-TB,2)/np.linalg.norm(T_newB+TB,2)
    
    resPA = np.linalg.norm(P_newA-PA,2)/np.linalg.norm(P_newA+PA,2)
    resPB = np.linalg.norm(P_newB-PB,2)/np.linalg.norm(P_newB+PB,2)
    
    res = max(resA, resB, resPA, resPB)
    
    TA = T_newA
    TB = T_newB
    PA = P_newA
    PB = P_newB
    print("Converged adiabatic profiles with max residual: %.3e"%res)    










# Plot the results 

def adiabatic_profile(z, T, rho, alpha, Cp, method_name, filename):
    fig, ax1 = plt.subplots(figsize=(8,6))
    
    color = 'tab:red'
    ax1.set_xlabel('Depth [km]')
    ax1.set_ylabel('Temperature [K]', color=color)
    ax1.plot(z/1e3, T, color=color, label='Temperature')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.invert_yaxis()
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('Density [kg/m3]', color=color)  # we already handled the x-label with ax1
    ax2.plot(z/1e3, rho, color=color, label='Density')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(f'Adiabatic Profile using {method_name}')
    plt.grid(alpha=0.3, linestyle=':')
    plt.show()
    fig.savefig(filename, dpi=300)


def plot_map_P_T(X, Y, Z, title, xlabel, ylabel, cbar_label, filename):
    fig = plt.figure(figsize=(8,6))
    ax = plt.gca()
    a0 = ax.contourf(X-273.15,Y/1e9, Z, levels=10, cmap='cmc.lipari')
    ax.yaxis_inverted()
    plt.colorbar(a0,label=cbar_label)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()  
    fig.savefig(filename, dpi=300)
    
def error_plot(X,B,name,xl,yl,nm):
    fg = plt.figure(figsize=(8,6))
    ax = fg.gca()
    ax.plot(X, B,c='firebrick')
    ax.legend()
    ax.set_ylabel(xl)
    ax.set_xlabel(yl)
    ax.set_title(nm)
    ax.grid(alpha=0.3, linestyle=':')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(direction='in', which='both', length=6, width=1, colors='black', grid_color='black', grid_alpha=0.5)
    ax.tick_params(axis='x', direction='in', which='both', bottom=True, top=False)
    ax.tick_params(axis='y', direction='in', which='both', left=True, right=False)


    fg.savefig('%s.png'%name, dpi=300)
    

import cmcrameri as cm


plot_map_P_T(T, P, k+k_rad, r'$k$ as a function of Pressure and Temperature', 'Temperature (C)', 'Pressure (GPa)', 'K [W/m/K]', 'thermal_conductivity.png')

plot_map_P_T(T, P, rho, r'$\rho_{A}$ as a function of Pressure and Temperature (Method A)', 'Temperature (C)', 'Pressure (GPa)', r'$\rho^{A}$ [kg/m3]', 'density_A.png')

plot_map_P_T(T, P, rho2, r'$\rho_{B}$ as a function of Pressure and Temperature (Method B)', 'Temperature (C)', 'Pressure (GPa)', r'$\rho^{B}$ [kg/m3]', 'density_B.png')

error_rho = (rho - rho2)/(rho + rho2)

plot_map_P_T(T, P, error_rho, r'Error in density between Method A and Method B', 'Temperature (C)', 'Pressure (GPa)', r'$\frac{\rho^{A}-\rho^{B}}{\rho^{A}+\rho^{B}}$', 'error_density.png')

plot_map_P_T(T, P, alpha_total, r'$\alpha$ as a function of Pressure and Temperature (Method A)', 'Temperature (C)', 'Pressure (GPa)', r'$\alpha^{A}$', 'alpha_A.png')

plot_map_P_T(T, P, alpha_total2, r'$\alpha$ as a function of Pressure and Temperature (Method B)', 'Temperature (C)', 'Pressure (GPa)', r'$\alpha^{B}$', 'alpha_B.png')

alpha_err = (alpha_total - alpha_total2)/(alpha_total + alpha_total2)

plot_map_P_T(T, P, alpha_err, r'Error in thermal expansivity between Method A and Method B', 'Temperature (C)', 'Pressure (GPa)', r'$\frac{\alpha^{A}-\alpha^{B}}{\alpha^{A}+\alpha^{B}}$', 'error_alpha.png')

plot_map_P_T(T, P, Cp, r'$C_p$ as a function of Pressure and Temperature', 'Temperature (C)', 'Pressure (GPa)', r'$C_p$ [J/kg/K]', 'Cp.png')

alpha_err = (alpha_total - alpha_total2)/(alpha_total + alpha_total2)

error_plot(P/1e9, alpha_err , 'alpha_error','Pressure [GPa]',r'err(\alpha)','Thermal Expansivity Error between Method A and B')

adiabatic_profile(z, TA, rho_A, alpha_A, CpA, 'Method A', 'adiabatic_profile_A.png')

adiabatic_profile(z, TB, rho_B, alpha_B, CpB, 'Method B', 'adiabatic_profile_B.png')

T_err = (TA - TB)/(TA + TB)

error_plot(z/1e3, T_err , 'temperature_error','Depth [km]','err(T_{adiabatic})','Temperature Error between Method A and B')

rho_err = (rho_A - rho_B)/(rho_A + rho_B)

error_plot(z/1e3, rho_err , 'density_error','Depth [km]',r'err(\rho_{adiabatic})','Density Error between Method A and B')
