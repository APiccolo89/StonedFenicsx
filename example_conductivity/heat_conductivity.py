import numpy as np 
import matplotlib.pyplot as plt 
'''
if option_k == 1 or option_k == 3: 
    k    = 4.10 
    n    = 0.493   
    a    = 0.032/1e9#[Convert from 1/GPa to 1/Pa]
elif option_k == 2 or option_k == 3 :
    k  = 2.47 # [Wm-1K-1]
    a = 0.33/1e9 # [Wm-1K-1GPa-1] GPA PD!!!!
    n  = 0.48 
    
    Script description: This script computes the density of a material as a function of pressure and temperature
using a combined equation of state approach based on the Birch-Murnaghan equation of state
and thermal expansivity models from Grose and Alfonso (2013). 
The idea is to understand how to compute these properties and giving a general framework for future implementation and starting 
a work flow that begins with the thermal structure of the oceanic lithosphere. 

I need a suitable material property set for the oceanic lithosphere, e.g., peridotite or basalt.
Then compute the density profile as a function of pressure and temperature. 
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

def compute_thermal_expansitivity_temperature_dependent(T, alpha0, alpha1 , T0 , q):
    
    alphaT = alpha0 * (T - T0)  + 0.5 * alpha1 * (T**2 - T0**2) 

    return alphaT
    

T = np.linspace(298.15, 1600+273.15, 1001) #[K]

k_rad = np.zeros(len(T))

k_rad = compute_radiative_conductivity_T(T,0.5)

P = np.linspace(0, 12e9, 600) #[Pa]

K0 = 130e9 #[Pa]

K0p = 4.8  

V_V0 = np.zeros(len(P))

alpha_p = np.zeros(len(P))

for i in range(len(P)):

    V_V0[i] = compute_isothermal_volume_change(P[i], K0, K0p)
    
    alpha_p[i] = compute_thermal_expansivity_pressure_dependent(V_V0[i], Gr0=6.0)


alpha_T = compute_thermal_expansitivity_temperature_dependent(T, alpha0=2.832e-5,alpha1 = 0.758e-8 , T0=273.15, q=1.0)

rho_0 = 3330.0 #[kg/m3]

rho    = np.zeros((len(P), len(T)))
rho2   = np.zeros((len(P), len(T)))
Kb     = (2*100e9*(1+0.25))/(3*(1-0.25*2))
alpha0 =  2.832e-5
alpha1 =  3.79e-8 #[1/K2]
k      = np.zeros((len(P), len(T)))
k_rad  = np.zeros((len(P), len(T))) 
Cp  = np.zeros((len(P), len(T))) 
kp = np.zeros((len(P), len(T)))


for i in range(len(P)):
    for j in range(len(T)):
        
        rho[i,j]   = rho_0 * V_V0[i] * (1-alpha_p[i] * alpha_T[j])
        rho2[i,j]  = rho_0 * np.exp(P[i]/Kb) * (1 - (alpha0 * (T[j]-273.15) + 0.5 * alpha1 * (T[j]**2 - 273.15**2)))
        k[i,j]     = compute_diffusivity_T(P[i], T[j]) * rho[i,j] * compute_thermal_capacity_T(T[j]) * compute_diffusivity_P(P[i])
        k_rad[i,j] = compute_radiative_conductivity_T(T[j],0.5)
        Cp[i,j]    = compute_thermal_capacity_T(T[j])
        kp[i,j]    = compute_diffusivity_P(P[i])
        


import cmcrameri as cm

fig = plt.figure(figsize=(8,6))
ax = plt.gca()
a0 = ax.contourf(T-273.15,P/1e9, (k+k_rad), levels=10, cmap='cmc.lipari')
b0 = ax.contour(T-273.15,P/1e9, (k+k_rad), levels=np.linspace(1,10,num=10), colors='w', linewidths=0.5)
ax.clabel(b0, inline=1, fontsize=10, fmt='%1.1f')
ax.yaxis_inverted()
plt.colorbar(a0,label='K (W/m/K)')
plt.ylabel('Pressure (GPa)')
plt.xlabel('Temperature (C)')
plt.title('Conductivity as a function of Pressure and Temperature')
plt.show()  
fig.savefig('thermal_conductivity_total.png', dpi=300)


fig = plt.figure(figsize=(8,6))
ax = plt.gca()
a0 = ax.contourf(T-273.15,P/1e9, (rho), levels=10, cmap='cmc.lipari')
b0 = ax.contour(T-273.15,P/1e9, (rho), levels=np.linspace(1,10,num=10), colors='w', linewidths=0.5)
ax.clabel(b0, inline=1, fontsize=10, fmt='%1.1f')
ax.yaxis_inverted()
plt.colorbar(a0,label='K (W/m/K)')
plt.ylabel('Pressure (GPa)')
plt.xlabel('Temperature (C)')
plt.title('Density as a function of Pressure and Temperature')
plt.show()  
fig.savefig('Density.png', dpi=300)


fig = plt.figure(figsize=(8,6))
ax = plt.gca()
a0 = ax.contourf(T-273.15,P/1e9, (rho-rho2), levels=10, cmap='cmc.lipari')
b0 = ax.contour(T-273.15,P/1e9, (rho-rho2), levels=np.linspace(1,10,num=10), colors='w', linewidths=0.5)
ax.clabel(b0, inline=1, fontsize=10, fmt='%1.1f')
ax.yaxis_inverted()
plt.colorbar(a0,label='K (W/m/K)')
plt.ylabel('Pressure (GPa)')
plt.xlabel('Temperature (C)')
plt.title('Density as a function of Pressure and Temperature')
plt.show()  
fig.savefig('Density.png', dpi=300)

plt.figure(figsize=(8,6))
plt.contourf(T-273.15,P/1e9, (Cp), levels=10, cmap='gist_heat')
plt.colorbar(label='Cp (J/kg/K)')
plt.ylabel('Pressure (GPa)')
plt.xlabel('Temperature (C)')
plt.title('Conductivity as a function of Pressure and Temperature')
plt.show()  

plt.figure(figsize=(8,6))
plt.contourf(T-273.15,P/1e9, (rho), levels=10, cmap='viridis')
plt.colorbar(label='Density (kg/m3)')
plt.ylabel('Pressure (GPa)')
plt.xlabel('Temperature (C)')
plt.title('Density as a function of Pressure and Temperature')
plt.show()