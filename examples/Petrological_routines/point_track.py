import numpy as np 
import h5py as hpy 
from dataclasses import dataclass
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

class PseudoSection:
    def __init__(self,path):
        self.P = None 
        self.T = None 
        self.rho = None
        self.water = None
        self.water2 = None
        self.inter_water = None
        self.rho_interp = None
        self.load_data(path)
    def load_data(self,path):
        with open(path, newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            data = list(reader)
            T_ph = []
            P_ph = []
            water = []
            water2 = []
            rho = []
            for i in data[13:]:
                values = i[0].split()
                T_ph.append(float(values[0]))
                P_ph.append(float(values[1])*1e5)
                water.append(float(values[2]))
                water2.append(float(values[3]))
                rho.append(float(values[4]))
                


        self.P=np.reshape(P_ph,(313,313))
        self.T=np.reshape(T_ph,(313,313))
        self.water = np.reshape(water,(313,313))
        self.water2 = np.reshape(water2,(313,313))
        self.rho = np.reshape(rho,(313,313))
        self.inter_water = RegularGridInterpolator((self.T[0,:],self.P[:,0]),self.water,method='nearest',bounds_error=False,fill_value=None)
        self.rho_interp = RegularGridInterpolator((self.T[0,:],self.P[:,0]),self.rho,method='nearest',bounds_error=False,fill_value=None)
        

class Points:
    def __init__(self,x0,y0,ID,phase):
        self.ID = ID
        self.phase = phase
        self.comp = 'Basalt'
        self.x = x0
        self.y = y0
        self.P = np.zeros([len(x0)],dtype=np.float64)
        self.T = np.zeros([len(x0)],dtype=np.float64)
        self.v = np.zeros([len(x0),2],dtype=np.float64)
        self.water_loss :float = None 
        self.rho :float = None 
        self.drho:float = None 
    
    def interpolatePT(self, X,Y,Pmod,Tmod):
        pass 
    
    def interpolateVel(self,vx,vy): 
        pass 
    
    def update_position(self,dt):
        # -> to update to Runge Kutta 
        self.x = self.x + self.vx * dt 
        self.y = self.y + self.vy * dt
    
    def update_composition(self,ctrls):
        # Run Perplex 
        # Check if the water is higher than    
        # If so -> remove -> recompute the composition
        # -> compute drho with the previous state 
        # -> run again perplex 
        # -> update rho 
        
        pass

# open test [Steady state for now]

# Open a test: 
from data_extractor import Test 

import csv

Peridotite = PseudoSection('/Users/wlnw570/Work/Leeds/Fenics_tutorial/examples/Petrological_routines/mantle_h2O/mantle_tab_def.tab')
Basalt = PseudoSection('/Users/wlnw570/Work/Leeds/Fenics_tutorial/examples/Petrological_routines/basalt_h2O/basalt_tab_def.tab')

Tst = Test('/Users/wlnw570/Work/Leeds/Fenics_tutorial/examples/Mexico/Mexico_tmax30_vc4_SelfConsistent_pr2_wzWetQuartzite_disl_0')
# Extract Coordinate: 

X = Tst.MeshData.X 
# Extract velocity

vx = Tst.Data_raw.SteadyState.vx 
vy = Tst.Data_raw.SteadyState.vy 
# Extract T,P 

T = Tst.Data_raw.SteadyState.Temp 
P = Tst.Data_raw.SteadyState.LitPres 

n_points = 30 

# initial position 
x = np.zeros([n_points,],dtype=np.float64)
y = np.zeros([n_points,],dtype=np.float64)

max_depth = -25.0 

y = np.linspace(0,max_depth,30)

phase = np.zeros([n_points,],dtype=np.int32)
phase[y>-6]=0
phase[y<=-6]=1 
ID = np.arange(0,30)
Pt = Points(x,y,ID,phase)


interp_vx = LinearNDInterpolator(X, vx)
interp_vy = LinearNDInterpolator(X, vy)
interp_P = LinearNDInterpolator(X, P)
interp_T = LinearNDInterpolator(X, T)

t = 0.0
dt = 0.001 * 365.25 * 24 * 60 * 60 * 1e6

for i in range(n_points):
    Pt.T[i] = interp_T(Pt.x[i],Pt.y[i])
    Pt.P[i] = interp_P(Pt.x[i],Pt.y[i])

fig = plt.figure()
ax = fig.gca()



while t < 10e6 * 365.25 * 24 * 60 * 60:
    water = np.zeros([n_points,],dtype=np.float64)
    for i in range(n_points):
        Pt.v[i,0] = interp_vx(Pt.x[i],Pt.y[i])*0.01/(365.25 * 24 * 60 * 60)
        Pt.v[i,1] = interp_vy(Pt.x[i],Pt.y[i])*0.01/(365.25 * 24 * 60 * 60)
        dx = (Pt.v[i,0] * dt)/1e3
        dy = (Pt.v[i,1] * dt)/1e3 
        Pt.x[i] = Pt.x[i] + dx 
        Pt.y[i] = Pt.y[i] + dy 
        Pt.T[i] = interp_T(Pt.x[i],Pt.y[i])+273.15
        Pt.P[i] = interp_P(Pt.x[i],Pt.y[i])*1e9
        water[i] = inter_phase((Pt.T[i],Pt.P[i]))

    
    a=ax.scatter(Pt.x,Pt.y,c=water,vmin=0,vmax=3)
    t = t + dt 
    print(f' t = {t/(365.25 * 24 * 60 * 60 * 1e6):.3f} Myr')

plt.colorbar(a,ax=ax)

        
        
        
    





# Construct points list: 










