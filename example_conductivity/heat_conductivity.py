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
'''

k0 = 4.10 
a = 0.032/1e9
n = 0.493

k1  = 2.47 # [Wm-1K-1]
a1 = 0.033/1e9 # [Wm-1K-1GPa-1] GPA PD!!!!
n1  = 0.48 


alpha = 1e-5  # thermal expansivity 1/K
T_ref = 273.15 + 25.0  # K
beta = 2e-12  # compressibility 1/Pa
Cp = 1000  # J/kgK




T0 = 298.0
P = np.linspace(0,35e3*3300*9.81,1000)
T = np.linspace(273.15,1100, 1000) # Pa
PP,TT = np.meshgrid(P,T)
rho = 2700*np.exp(-alpha*(TT - 273.15))*np.exp(beta*PP)  # kg/m3

K1 = k0 * (1 + a*PP)*(T0/TT)**n
K2 = k1 * (1 + a1*PP)*(T0/TT)**n1
tk = 5e3 
td1 = tk**2/(K1/(rho*Cp))/365.25/24/60/60/1e6  # in Myr
td2 = tk**2/(K2/(rho*Cp))/365.25/24/60/60/1e6  # in M

fig, ax = plt.subplots(1,3, figsize=(15,5))
cs1 = ax[0].contourf(PP/1e9, TT-273.15, K1, levels=50, cmap='viridis')
fig.colorbar(cs1, ax=ax[0])
ax[0].set_title('Conductivity model 1')
ax[0].set_xlabel('Pressure (GPa)')
ax[0].set_ylabel('Temperature (C)')
cs2 = ax[1].contourf(PP/1e9, TT-273.15, K2, levels=50, cmap='viridis')
fig.colorbar(cs2, ax=ax[1])
ax[1].set_title('Conductivity model 2')
ax[1].set_xlabel('Pressure (GPa)')
ax[1].set_ylabel('Temperature (C)')
cs3 = ax[2].contourf(PP/1e9, TT-273.15, td1 - td2, levels=50, cmap='bwr')
fig.colorbar(cs3, ax=ax[2])
ax[2].set_title('Difference in thermal diffusion time (Myr)')
ax[2].set_xlabel('Pressure (GPa)')
ax[2].set_ylabel('Temperature (C)')
plt.tight_layout()
plt.show()

Cp = 0.0*T 

Cp[T<846]  = (199.5 * 0.0857 * T[T<846])# - 5.0e6 * T[T<846] **(-2))
Cp[T>=846] = (229 * 0.0323 * T[T>=846])# - 47.9e6 * T[T>=846] **(-2))