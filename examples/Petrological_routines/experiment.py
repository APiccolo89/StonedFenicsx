import h5py
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

f = h5py.File('/Users/wlnw570/Work/Leeds/Fenics_tutorial/examples/Petrological_routines/data.h5', 'r')
x = np.array(f['x'])
y = np.array(f['y'])
P = np.array(f['P'])
T = np.array(f['T'])
fl_ex = np.array(f['fl_ex'])
H2O = np.array(f['H2O'])
f.close()
f = h5py.File('/Users/wlnw570/Work/Leeds/Fenics_tutorial/Basalt_phase_diagram.h5', 'r')
T_grid = np.array(f['T'])
P_grid = np.array(f['P'])
fl_grid = np.array(f['fl_grid'])
H2O_grid = np.array(f['H2O_grid'])
rho_grid = np.array(f['rho_grid'])
f.close()

fig = plt.figure()
ax = fig.gca()
a = ax.contourf(T_grid, P_grid, H2O_grid, levels = 20)
a2 = ax.scatter(T,P/10,c='r',s=0.8,alpha=0.5)
ax.set_ylim(0,8.0)
ax.set_xlim(0,1200)
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Pressure (GPa)')
plt.colorbar(a,ax=ax)


fig = plt.figure()
ax = fig.gca()
a = ax.contourf(T_grid, P_grid, H2O_grid, levels = 20)
plt.colorbar(a,ax=ax)







fig = plt.figure()
ax = fig.gca()
a=ax.scatter(x,y,c=np.log10(H2O),s=0.8,vmin = np.log10(0.008),vmax = np.log10(0.031))
plt.colorbar(a,ax=ax,label='log_10(H2O) content (wt%)')
ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_title('H2O content (wt%)')
ax.set_ylim(-200,0)
ax.set_xlim(0,700)

print('done')

fig = plt.figure()
ax = fig.gca()
a=ax.scatter(x,y,c=T,s=0.8,vmin = 200,vmax = 1000)
print('done')

fig = plt.figure()
ax = fig.gca()
a=ax.scatter(x,y,c=P/10,s=0.8,vmin = 1,vmax = 10)
print('done')