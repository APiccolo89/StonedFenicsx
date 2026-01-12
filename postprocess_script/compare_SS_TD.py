import vtk 
from mpi4py import MPI
import numpy as np
import dolfinx 
import h5py
from dolfinx.io import XDMFFile
import os
import cmcrameri
from data_extractor import Test
from data_extractor import MeshData

def release_vectorial_field(vec_field,T,ind_sortS,ind_sortO):
    
    if vec_field == 'HeatFlux':
        vx = 'qx'
        vy = 'qy'
    else:
        vx = 'vx'
        vy = 'vy'
    
    ax    = eval('T.Data_raw.SteadyState.%s[M_data.ind_topSlab]'%vx)
    ay    = eval('T.Data_raw.SteadyState.%s[M_data.ind_topSlab]'%vy)
    
    ox   = eval('T.Data_raw.SteadyState.%s[M_data.ind_Oceanic]'%vx)
    oy   = eval('T.Data_raw.SteadyState.%s[M_data.ind_Oceanic]'%vy)
    
    s_slab = np.sqrt(ax**2 + ay**2)
    s_oc   = np.sqrt(ox**2 + oy**2)
    
    s_slab = s_slab[ind_sortS]
    s_oc   = s_oc[ind_sortO]
    
    return s_slab,s_oc

def release_scalar_field(scalar_field,T,ind_sortS,ind_sortO):
    
    s_slab = eval('T.Data_raw.SteadyState.%s[M_data.ind_topSlab]'%scalar_field)
    s_oc   = eval('T.Data_raw.SteadyState.%s[M_data.ind_Oceanic]'%scalar_field)

    s_slab = s_slab[ind_sortS]
    s_oc   = s_oc[ind_sortO] 

    return s_slab,s_oc
    

def compare_slab_surface(path2save:str,
                        time_string:str,
                        ipic:int,
                        XO:float,
                        XS:float,
                        QO:float,
                        QS:float,
                        S:float,
                        OC:float,
                        string_label:list,
                        fname:str='comparison',
                        ind:int = [-1,-1]):
    from matplotlib import pyplot as plt
    
    plt.rcParams.update({
    "text.usetex": True,           # Use LaTeX for all text
    "font.family": "serif",        # Or 'sans-serif'
    "font.serif": ["Computer Modern"],   # LaTeX default
    "axes.unicode_minus": False,
    })
    
    
    def modify_ax(ax,fg):
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(direction='in', which='both', length=6, width=1, colors='black', grid_color='black', grid_alpha=0.5)
        ax.tick_params(axis='x', direction='in', which='both', bottom=False, top=True)
        ax.tick_params(axis='y', direction='in', which='both', left=True, right=False)
        #ax = arrowed_spines(fg, ax)
        ax.set_ylim([800,0.0])
        return ax 
    
    color = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#F0E442",  # yellow
    "#56B4E9",  # light blue
    "#E69F00",  # orange
    "#000000",  # black
    "#999999",  # grey
    "#332288",  # dark blue / indigo
    ]
    fname = os.path.join(path2save, 'Slab_surface_comparison%s'%fname)
    if not os.path.isdir(fname):
        os.makedirs(fname)
    figure_name = f'Figure_{ipic:03d}.png'
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(10,6))
    
    bx = 0.08 
    
    by = 0.04
    
    dy = 0.08 
    
    dx = 0.01 
    
    sx = 0.40
    
    sy = 0.4 
    
    ax0 = fig.add_axes([bx, by+sy+dy, sx, sy])
    
    ax1 = fig.add_axes([bx+sx+dx, by+sy+dy, sx, sy])
    
    ax2 = fig.add_axes([bx, by, sx, sy])
    
    ax3 = fig.add_axes([bx+sx+dx, by, sx, sy])
    
    for i in range(len(S)):
        ax0.plot(S[i],XS,c=color[i],label=r'$T^{\infty}$%s Slab'%string_label[i] ,linewidth=1.0,linestyle='-.'                )

    for i in range(len(OC)):
        ax2.plot(OC[i],XO,c=color[i],label=r'$T^{\infty}$%s Oceanic Crust'%string_label[i] ,linewidth=1.0,linestyle='-.'                )
    
    for i in range(len(QS)):
        ax1.plot(QS[i],XS,c=color[i],label=r'$q^{\infty}$%s Slab'%string_label[i] ,linewidth=1.0,linestyle='-.'                )
    
    for i in range(len(QO)): 
        ax3.plot(QO[i],XO,c=color[i],label=r'$q^{\infty}$%s Oceanic Crust'%string_label[i] ,linewidth=1.0,linestyle='-.'                )
    
    ax2.set_xlabel(r'T [$^{\circ} C$]', fontsize=12)
    
    ax0.set_ylabel(r'$\ell_{sl}$ [km]', fontsize=12)
    
    ax2.set_ylabel(r'$\ell_{oc}$ [km]', fontsize=12)
    
    ax3.set_xlabel(r'q [$W/m^2$]', fontsize=12)
    
    ax1.set_yticklabels([])
    
    ax3.set_yticklabels([])


    ax0 = modify_ax(ax0,fig)
    
    ax1 = modify_ax(ax1,fig)
    
    ax2 = modify_ax(ax2,fig)
    
    ax3 = modify_ax(ax3,fig)
    
    ax0.legend(fontsize=8, loc='lower left', frameon=False)
    ax1.legend(fontsize=8, loc='lower right', frameon=False)


    ax2.legend(fontsize=8, loc='lower left', frameon=False)
    ax3.legend(fontsize=8, loc='lower right', frameon=False)
    
    ax0.xaxis.set_label_position("top")
    
    ax1.xaxis.set_label_position("top")
    
    ax2.xaxis.set_label_position("top")
    
    ax3.xaxis.set_label_position("top")
    
    ax0.grid(True, linestyle='-.', linewidth=0.2)
    
    ax1.grid(True, linestyle='-.', linewidth=0.2)

    ax2.grid(True, linestyle='-.', linewidth=0.2)
    
    ax3.grid(True, linestyle='-.', linewidth=0.2)
    
    if ind[0] !=0: 
        ax1.axhline(XS[ind[0]],linewidth= 1.0,  alpha=0.7)
        ax0.axhline(XS[ind[0]],linewidth = 1.0, alpha=0.7)
        ax2.axhline(XO[ind[1]],linewidth = 1.0, alpha=0.7)
        ax3.axhline(XO[ind[1]],linewidth = 1.0, alpha=0.7)
        ax1.axhline(XS[ind[2]],linewidth= 1.0,  alpha=0.7)
        ax0.axhline(XS[ind[2]],linewidth = 1.0, alpha=0.7)
        ax2.axhline(XO[ind[3]],linewidth = 1.0, alpha=0.7)
        ax3.axhline(XO[ind[3]],linewidth = 1.0, alpha=0.7)

        
    
    props = dict(boxstyle='round', facecolor='black', alpha=0.5)
    
    
    ax0.text(0.9, 1.15, '[a]', transform=ax0.transAxes, fontsize=8,
        verticalalignment='top', bbox=props,color='white')
    ax1.text(0.9, 1.15,'[b]', transform=ax1.transAxes, fontsize=8,
        verticalalignment='top', bbox=props,color='white')
    ax2.text(0.9, 1.15, '[c]', transform=ax2.transAxes, fontsize=8,
        verticalalignment='top', bbox=props,color='white')
    ax3.text(0.9, 1.15, '[d]', transform=ax3.transAxes, fontsize=8,
        verticalalignment='top', bbox=props,color='white')
 
    ax0.text(0.1, 1.15, time_string, transform=ax0.transAxes, fontsize=10,
        verticalalignment='top', bbox=props,color='white')
 
    fig.savefig(os.path.join(fname, figure_name))

    return 0



def create_figure(path2save:str, 
                  M_data:MeshData,
                  vmin:float, 
                  vmax:float, 
                  cmap: str, 
                  title: str,
                  field:float, 
                  n_level:int,
                  name_fig:str,
                  ipic:int,
                  name_field:str): 
    
    import matplotlib.pyplot as plt

    plt.rcParams.update({
    "text.usetex": True,           # Use LaTeX for all text
    "font.family": "serif",        # Or 'sans-serif'
    "font.serif": ["Computer Modern"],   # LaTeX default
    "axes.unicode_minus": False,
    })
    def modify_ax(ax):
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
        return ax

    xs = M_data.X[(M_data.mesh_tag==8.0) | (M_data.mesh_tag == 9.0),0]
    ys = M_data.X[(M_data.mesh_tag==8.0) | (M_data.mesh_tag == 9.0),1]
    sort = np.argsort(xs)
    
    x_s  = [xs[sort], ys[sort]]
    
    xor = M_data.X[(M_data.mesh_tag==11.0),0]
    yor = M_data.X[(M_data.mesh_tag==11.0),1]
    sort = np.argsort(xor)
    
    x_or  = [xor[sort], yor[sort]]
    
    xbt = M_data.X[(M_data.mesh_tag==6.0),0]
    ybt = M_data.X[(M_data.mesh_tag==6.0),1]
    sort = np.argsort(xbt)
    
    x_bt  = [xbt[sort], ybt[sort]]
    
    
    
    pt_save = os.path.join(path2save,name_fig)
    if not os.path.isdir(pt_save):
        os.makedirs(pt_save)
        
    figure_name = f'Figure_{ipic:03d}_%s.png'%name_field
        
    fname = os.path.join(pt_save, figure_name)
    fig, ax0 = plt.subplots(figsize=(10, 6))
    ax0 = modify_ax(ax0)
    ax0.set_title(time_string, fontsize=16)
    ax0.set_xlabel('Distance [km]', fontsize=14)
    ax0.set_ylabel('Depth [km]', fontsize=14)
    p0 = ax0.contourf(M_data.Xi, M_data.Yi, field, levels=n_level, cmap=cmap, vmin=vmin, vmax=vmax)
    p1 = ax0.plot(x_s[0],x_s[1],c='w',linewidth=1.2)
    p2 = ax0.plot(x_or[0],x_or[1],c='w',linewidth=1.2)
    p3 = ax0.plot(x_bt[0],x_bt[1],c='k',linewidth=1.2)
    cbar = plt.colorbar(p0, ax=ax0, orientation='vertical', pad=0.02)
    cbar.set_label(label=title, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    #cbar.set_ticks(np.linspace(vmin, vmax, 5))
    #cbar.set_ticklabels([f'{val:.0f}' for val in np.linspace(vmin, vmax, 5)])
    fig.savefig(fname)
    plt.close(fig)
    
    return 0 




path = '/Users/wlnw570/Work/Results'
path2save = '/Users/wlnw570/Work/Figures_Tests'
if not os.path.isdir(path2save):
    os.makedirs(path2save)  
    

T0 = Test('%s/exp0_bis/Output'%path)
T1 = Test('%s/exp1_bis/Output'%path)
T2 = Test('%s/exp2_bis/Output'%path)
T3 = Test('%s/exp3_bis/Output'%path)
T4 = Test('%s/exp4_bis/Output'%path)
T5 = Test('%s/exp4_bisNS/Output'%path)

Temp_T0 = T0._interpolate_data('SteadyState.Temp')
Temp_T1 = T1._interpolate_data('SteadyState.Temp')
Temp_T2 = T2._interpolate_data('SteadyState.Temp')
Temp_T3 = T3._interpolate_data('SteadyState.Temp')
Temp_T4 = T4._interpolate_data('SteadyState.Temp')
Temp_T5 = T5._interpolate_data('SteadyState.Temp')

rho0 = T0._interpolate_data('SteadyState.Temp')
rho1 = T1._interpolate_data('SteadyState.Temp')
rho2 = T2._interpolate_data('SteadyState.Temp')
rho3 = T3._interpolate_data('SteadyState.Temp')
rho4 = T4._interpolate_data('SteadyState.Temp')
rho5 = T5._interpolate_data('SteadyState.Temp')

Cp0 = T0._interpolate_data('SteadyState.Temp')
Cp1 = T1._interpolate_data('SteadyState.Temp')
Cp2 = T2._interpolate_data('SteadyState.Temp')
Cp3 = T3._interpolate_data('SteadyState.Temp')
Cp4 = T4._interpolate_data('SteadyState.Temp')
Cp5 = T5._interpolate_data('SteadyState.Temp')

alpha0 = T0._interpolate_data('SteadyState.Temp')
alpha1 = T1._interpolate_data('SteadyState.Temp')
alpha2 = T2._interpolate_data('SteadyState.Temp')
alpha3 = T3._interpolate_data('SteadyState.Temp')
alpha4 = T4._interpolate_data('SteadyState.Temp')
alpha5 = T5._interpolate_data('SteadyState.Temp')


M_data = T0.MeshData

vmin = 0.0
vmax = 1700.0 

n_level = 100
cmap = cmcrameri.cm.lipari
title = 'Temperature [$^{\circ}C$]'
name_fig = 'Temperature'
time_string = 'Steady State'
create_figure(path2save,M_data,vmin,np.nanmax(Temp_T0),cmap,title,Temp_T0,n_level,name_fig,0,'e_0_NA_B')
create_figure(path2save,M_data,vmin,np.nanmax(Temp_T1),cmap,title,Temp_T1,n_level,name_fig,0,'e_1_YA_B')
create_figure(path2save,M_data,vmin,np.nanmax(Temp_T2),cmap,title,Temp_T2,n_level,name_fig,0,'e_2_YA_NL_B')
create_figure(path2save,M_data,vmin,np.nanmax(Temp_T3),cmap,title,Temp_T3,n_level,name_fig,0,'e_3_YA_NL_C_B')   
create_figure(path2save,M_data,vmin,np.nanmax(Temp_T4),cmap,title,Temp_T4,n_level,name_fig,0,'e_4_YA_NL_C_RB')   
create_figure(path2save,M_data,vmin,np.nanmax(Temp_T5),cmap,title,Temp_T5,n_level,name_fig,0,'e_4_YA_NL_C_RB')   

# --- 



# Difference T1 - T0

vmin = np.floor(np.nanmin(Temp_T1 - Temp_T0))
vmax = np.ceil(np.nanmax(Temp_T1 - Temp_T0))
n_level = 100
cmap = cmcrameri.cm.vik
title = 'Temperature Difference e_1_YA - e_0_NA [$^{\circ}C$]'
name_fig = 'Temperature_difference_e_1_minus_e_0'
create_figure(path2save,M_data,vmin,vmax,cmap,title,Temp_T1-Temp_T0,n_level,name_fig,0,'e_1_minus_e_0Bis')

# Difference T2 - T0
vmin = np.floor(np.nanmin(Temp_T2 - Temp_T1))
vmax = np.ceil(np.nanmax(Temp_T2 - Temp_T1))
n_level = 100
cmap = cmcrameri.cm.vik
title = 'Temperature Difference e_2_YA_NL - e_1_YA [$^{\circ}C$]'
name_fig = 'Temperature_difference_e_2_minus_e_1'
create_figure(path2save,M_data,vmin,vmax,cmap,title,Temp_T2-Temp_T1,n_level,name_fig,0,'e_2_minus_e_1Bis')

# Difference T3 - T0
vmin = np.floor(np.nanmin(Temp_T3 - Temp_T1))
vmax = np.ceil(np.nanmax(Temp_T3 - Temp_T1))
n_level = 100
cmap = cmcrameri.cm.vik
title = 'Temperature Difference e_3_YA_NL_C - e_1_YA [$^{\circ}C$]'
name_fig = 'Temperature_difference_e_3_minus_e_1'
create_figure(path2save,M_data,vmin,vmax,cmap,title,Temp_T3-Temp_T1,n_level,name_fig,0,'e_3_minus_e_1Bis')

vmin = np.floor(np.nanmin(Temp_T4 - Temp_T1))
vmax = np.ceil(np.nanmax(Temp_T4 - Temp_T1))
n_level = 100
cmap = cmcrameri.cm.vik
title = 'Temperature Difference e_4_YA_NL_C - e_1_YA [$^{\circ}C$]'
name_fig = 'Temperature_difference_T0e_minus_T0b'
create_figure(path2save,M_data,vmin,vmax,cmap,title,Temp_T4-Temp_T1,n_level,name_fig,0,'e_4_minus_e_1Bis')

vmin = np.floor(np.nanmin(Temp_T5 - Temp_T4))
vmax = np.ceil(np.nanmax(Temp_T5 - Temp_T4))
n_level = 100
cmap = cmcrameri.cm.vik
title = 'Temperature Difference e_4_YA_NL_C_NS - e_4_YA_NL_C [$^{\circ}C$]'
name_fig = 'Temperature_difference_T0e_minus_T0b'
create_figure(path2save,M_data,vmin,vmax,cmap,title,Temp_T5-Temp_T4,n_level,name_fig,0,'e_4_minus_e_4NSBis')

x      = M_data.X[:,0]
y      = M_data.X[:,1]
xs_slab = x[M_data.ind_topSlab]
ys_slab = y[M_data.ind_topSlab]
xs_ocmoh = x[M_data.ind_Oceanic]
ys_ocmoh = y[M_data.ind_Oceanic]
sort_slab = np.argsort(np.abs(ys_slab))
sort_ocmoh = np.argsort(np.abs(ys_ocmoh))
ys_slab = ys_slab[sort_slab]
xs_slab = xs_slab[sort_slab]
xs_ocmoh = xs_ocmoh[sort_ocmoh]
ys_ocmoh = ys_ocmoh[sort_ocmoh]


length_slab = np.zeros(np.size(xs_slab),dtype = np.float64)
for i in range(len(xs_slab)): 
    if i > 0: 
        d = np.sqrt((xs_slab[i]-xs_slab[i-1])**2 + (ys_slab[i]-ys_slab[i-1])**2)
        length_slab[i] = length_slab[i-1]+d  

length_ocean = np.zeros(np.size(xs_ocmoh),dtype = np.float64)
for i in range(len(xs_ocmoh)): 
    if i > 0: 
        d = np.sqrt((xs_ocmoh[i]-xs_ocmoh[i-1])**2 + (ys_ocmoh[i]-ys_ocmoh[i-1])**2)
        length_ocean[i] = length_ocean[i-1]+d  
    

ind_dc_s = np.where(ys_slab==-80)[0]
ind_dc_o = np.where(ys_slab==-80)[0]
ind_dc_s1 = np.where(ys_slab<-50)[0][0]
ind_dc_o1 = np.where(ys_slab<-50)[0][0]
ind = [ind_dc_s,ind_dc_o,ind_dc_s1,ind_dc_o1]
T0S,O0S = release_scalar_field('Temp',T0,sort_slab,sort_ocmoh)
T1S,O1S = release_scalar_field('Temp',T1,sort_slab,sort_ocmoh)
T2S,O2S = release_scalar_field('Temp',T2,sort_slab,sort_ocmoh)
T3S,O3S = release_scalar_field('Temp',T3,sort_slab,sort_ocmoh)
T4S,O4S = release_scalar_field('Temp',T4,sort_slab,sort_ocmoh)
T5S,O5S = release_scalar_field('Temp',T5,sort_slab,sort_ocmoh)


qx0S,Oq0S = release_vectorial_field('HeatFlux',T0,sort_slab,sort_ocmoh)
qx1S,Oq1S = release_vectorial_field('HeatFlux',T1,sort_slab,sort_ocmoh)
qx2S,Oq2S = release_vectorial_field('HeatFlux',T2,sort_slab,sort_ocmoh)
qx3S,Oq3S = release_vectorial_field('HeatFlux',T3,sort_slab,sort_ocmoh)
qx4S,Oq4S = release_vectorial_field('HeatFlux',T4,sort_slab,sort_ocmoh)
qx5S,Oq5S = release_vectorial_field('HeatFlux',T5,sort_slab,sort_ocmoh)



TSlab = [T0S,T1S,T2S,T3S,T4S,T5S]
Tocmoh = [O0S,O1S,O2S,O3S,O4S,O5S]
q0s_slab = [qx0S,qx1S,qx2S,qx3S,qx4S,qx5S]
q0s_ocmoh = [Oq0S,Oq1S,Oq2S,Oq3S,Oq4S,Oq5S]

label = ['e_0_NAB','e_1_YAB','e_2_YA_NLB','e_3_YA_NL_CB','e_4_YA_NL_C_RB','e_4_YA_NL_C_RBNS']

compare_slab_surface(path2save,'',0,length_ocean,length_slab,q0s_ocmoh,q0s_slab,TSlab,Tocmoh,label,'',ind)


TSlab = [T2S,T3S,T4S] -T1S
Tocmoh = [O2S,O3S,O4S]-O1S
q0s_slab = [qx2S,qx3S,qx4S]-qx1S
q0s_ocmoh = [Oq2S,Oq3S,Oq4S]-Oq1S

label = ['e_2_YA_NLB','e_3_YA_NL_CB','e_4_YA_NL_C_RB']


compare_slab_surface(path2save,'',0,length_ocean,length_slab,q0s_ocmoh,q0s_slab,TSlab,Tocmoh,label,'difference',ind)









'''
class MeshData(): 
    def __init__(self,f:str,f1:str):
        # Extract mesh geometry 
        fl = h5py.File(f, 'r')

        X                        = np.array(fl['/Mesh/mesh/geometry'])
        
        self.X                   = X
        
        self.xi                  = np.linspace(np.min(X[:,0]),np.max(X[:,0]),2000)
        
        self.yi                  = np.linspace(np.min(X[:,1]),np.max(X[:,1]),2000)
        
        self.Xi,      self.Yi     = np.meshgrid(self.xi,self.yi)
        
        self.polygon, self.ar     = self.create_polygon()
        
        
        
        #self.subduction_arr     = 
        #self.crust_array        = 
        
        
        fl.close()
        fl = h5py.File(f1, 'r')
        ar_point = np.array(fl['Function/MeshTAG/0']) 
        ind = np.where(ar_point!=0.0)
        ind = ind[0]
        self.mesh_tag = ar_point.flatten()
        
        fl.close()
        

       
        
    def create_polygon(self):
        
        x_min = np.min(self.X[:,0])
        
        x_max = np.max(self.X[:,0])
        
        y_min = np.min(self.X[:,1])
        
        y_max = np.max(self.X[:,1])
        
        x = self.X[:,0]
        
        y = self.X[:,1]

        top    = np.array([x[self.X[:,1]==y_max],y[ self.X[:,1]==y_max]])
        
        bottom = np.array([x[self.X[:,1]==y_min],y[ self.X[:,1]==y_min]])
        
        left   = np.array([x[self.X[:,0]==x_min],y[ self.X[:,0]==x_min]])
        
        right  = np.array([x[self.X[:,0]==x_max],y[ self.X[:,0]==x_max]])
        
        x_bot  = np.min(bottom[0,:])
        
        l_min  = np.min(left[1,:])
        
        p0     = np.array([x_min, l_min])
        
        p_list = []
        
        p_list.append(p0)
        
        it     = 0
        
        p      = p0 
         
        while p[0] != x_bot and p[1] != y_min:
            """
            Ora, sono molto stupido, e lo riconosco, quindi ho costruito questo aborto usando un po' trial and error. Parto dal punto piu' basso a sinistra,
            da li vado a cercare i punti che hanno una distanza minore di 6 km, tra questi scelgo quelli che hanno il minore y del punto precedente, e poi seleziono
            le nuove coordinate del punto che verra' usato per la successiva iterazione. Correzione dovuta, arg sort in funzione di y. L'idea e' stata modificata 
            a tentativi, perche', ripeto, non sono una persona brillante e probabilmente avrei dovuto essere abortito prima di nascere. 
            """


            d = np.sqrt((x-p_list[it][0])**2 + (y-p_list[it][1])**2)
          
            idx = np.where((d < 6.0))   
          
            x_pr = x[idx[0]];y_pr = y[idx[0]];d_pr = d[idx[0]]
          
            x_pr = x_pr[y_pr<p_list[it][1]];d_pr = d_pr[y_pr<p_list[it][1]];y_pr = y_pr[y_pr<p_list[it][1]]
          
            d_pr_sort = np.argsort(y_pr)
          
            d_pr = d_pr[d_pr_sort];x_pr = x_pr[d_pr_sort];y_pr = y_pr[d_pr_sort]
          
            x_new = x_pr[0]
          
            y_new = y_pr[0]
          
            p = np.array([x_new,y_new])
          
            p_list.append(p)
                    
            it = it + 1 
            
        bottom = bottom.transpose()
        
        right = right.transpose()
        
        top = top.transpose()
        
        left = left.transpose()
        # order bottom boundary 
        
        x_bottom = bottom[:,0]
        
        ind_arg = np.argsort(x_bottom)
        
        bottom = bottom[ind_arg,:]
        # order right boundary
        
        y_right = right[:,1]
        
        ind_arg = np.argsort(y_right)
        
        right = right[ind_arg,:]
        # order top boundary
        
        x_top = top[:,0]
        
        ind_arg = np.argsort(x_top)
        
        top = top[np.invert(ind_arg),:]
        
        from shapely.geometry import Polygon as sPolygon
        
        from shapely import contains_xy as scontains_xy
        
        polygon = sPolygon(np.vstack((np.array(p_list), np.array(bottom), np.array(right), np.array(top), np.array(left))))
        
        ar =  scontains_xy(polygon,self.Xi,self.Yi)

        return polygon, ar 

def arrowed_spines(fig, ax):

    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)

    # removing the axis ticks
    #plt.xticks([]) # labels 
    #plt.yticks([])
    #ax.xaxis.set_ticks_position('none') # tick markers
    #ax.yaxis.set_ticks_position('none')

    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./20.*(ymax-ymin) 
    hl = 1./20.*(xmax-xmin)
    lw = 1. # axis line width
    ohg = 0.3 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
             head_width=hw, head_length=hl, overhang = ohg, 
             length_includes_head= True, clip_on = False) 

    ax.arrow(0, ymax, 0., ymin-ymax, fc='k', ec='k', lw = lw, 
             head_width=yhw, head_length=yhl, overhang = ohg, 
             length_includes_head= True, clip_on = False)
    return ax



def interpolate_field(field,M_Data): 
    """_summary_

    Args:
        field (float): field to interpolate into a regulard grid 
        M_Data (MeshData): data containing mesh information 

    Returns:
        field_i (float): interpolated field on regular grid 
        -> filtered such that only valuues inside the polygon are kept 
    """
    
    from scipy.interpolate import griddata

    field = field.flatten()
    # Interpolate field on regulard grid 
    field_i = griddata(M_data.X, field, (M_data.Xi, M_data.Yi), method='linear', fill_value=np.nan)
    # Filter the data
    field_i[M_data.ar==False] = np.nan
    
    return field_i

def compare_heatfluxes(path2save:str,
                        time_string:str,
                        ipic:int,
                        qsx:float,
                        qsy:float,
                        qsm:float,
                        qtdx:float,
                        qtdy:float,
                        qtdm:float,
                        M_data:MeshData):
    
    def modify_ax(ax,fg):
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(direction='in', which='both', length=6, width=1, colors='black', grid_color='black', grid_alpha=0.5)
        ax.tick_params(axis='x', direction='in', which='both', bottom=False, top=True)
        ax.tick_params(axis='y', direction='in', which='both', left=True, right=False)
        #ax = arrowed_spines(fg, ax)
        return ax 

            
    from matplotlib import pyplot as plt
    
    plt.rcParams.update({
        "text.usetex": True,           # Use LaTeX for all text
        "font.family": "serif",        # Or 'sans-serif'
        "font.serif": ["Computer Modern"],   # LaTeX default
        "axes.unicode_minus": False,
        })    
    
    ind_top  = ((M_data.mesh_tag==8.0) | (M_data.mesh_tag == 9.0)) & (M_data.X[:,1]>=-100.0)  
    z        = M_data.X[:,1]
    z_pl     = z[ind_top]
    sort     = np.argsort(z_pl)
    z_pl     = z_pl[sort]
    qsx      = qsx[ind_top][sort]
    qsy      = qsy[ind_top][sort]
    qsm      = qsm[ind_top][sort]
    qtdx     = qtdx[ind_top][sort]
    qtdy     = qtdy[ind_top][sort]
    qtdm     = qtdm[ind_top][sort]
      
    fname = os.path.join(path2save, 'Heatflux_comparison')
    
    if not os.path.isdir(fname):
        os.makedirs(fname)
    
    figure_name = f'Figure_{ipic:03d}.png'  
    
    fig = plt.figure(figsize=(10,6))
    
    bx = 0.1 
    by = 0.1
    
    dx = 0.05   
    
    sx = 0.25
    sy = 0.7
    
    ax0 = fig.add_axes([bx, by, sx, sy])
    ax1 = fig.add_axes([bx+sx+dx, by, sx, sy])
    ax2 = fig.add_axes([bx+2.0*(sx+dx), by, sx, sy])
    
    ax0.plot(qsx,  z_pl,c='forestgreen',label=r'$q_x^{\infty}$ Slab'                    ,linewidth=1.0, linestyle='-.')
    ax0.plot(qtdx, z_pl,c='firebrick',label=r'$q_x(t)$ Slab'     ,linewidth=1.2                )
    ax1.plot(qsy,  z_pl,c='forestgreen',label=r'$q_y^{\infty}$ Slab',linewidth=1.0,linestyle='-.'                )
    ax1.plot(qtdy, z_pl,c='firebrick',label=r'$q_y(t)$ Slab',        linewidth=1.2)
    ax2.plot(qsm,  z_pl,c='forestgreen',label=r'$|q|^{\infty}$ Slab',linewidth=1.0,linestyle='-.'                )
    ax2.plot(qtdm, z_pl,c='firebrick',label=r'$|q|(t)$ Slab',        linewidth=1.2)
    ax0.set_xlabel(r'$q_x$ [W/m$^2$]', fontsize=12)
    ax0.set_ylabel('Depth [km]', fontsize=12)
    ax1.set_yticklabels([])
    ax2.set_yticklabels([])
    ax1.set_xlabel(r'$q_y$ [W/m$^2$]', fontsize=12)
    ax2.set_xlabel(r'$|q|$ [W/m$^2$]', fontsize=12)
    ax0 = modify_ax(ax0,fig)
    ax1 = modify_ax(ax1,fig)
    ax2 = modify_ax(ax2,fig)
    ax0.legend(fontsize=8, loc='lower left', frameon=False)
    ax1.legend(fontsize=8, loc='lower left', frameon=False)
    ax2.legend(fontsize=8, loc='lower left', frameon=False)
    ax0.xaxis.set_label_position("top")
    ax1.xaxis.set_label_position("top")
    ax2.xaxis.set_label_position("top")
    ax0.grid(True, linestyle='-.', linewidth=0.2)
    ax1.grid(True, linestyle='-.', linewidth=0.2)
    ax2.grid(True, linestyle='-.', linewidth=0.2)
    props = dict(boxstyle='round', facecolor='black', alpha=0.5)
    ax0.text(0.9, 1.15, '[a]', transform=ax0.transAxes, fontsize=8,
        verticalalignment='top', bbox=props,color='white')
    ax1.text(0.9, 1.15,'[b]', transform=ax1.transAxes, fontsize=8,
        verticalalignment='top', bbox=props,color='white')
    ax2.text(0.9, 1.15, '[c]', transform=ax2.transAxes, fontsize=8,
        verticalalignment='top', bbox=props,color='white')
    ax0.text(0.1, 1.15, time_string, transform=ax0.transAxes, fontsize=10,
        verticalalignment='top', bbox=props,color='white')
    fig.savefig(os.path.join(fname, figure_name))
    
    
    return 0 

def compare_slab_surface(path2save:str,
                        time_string:str,
                        ipic:int,
                        Ts:float,
                        Ttd:float,
                        M_data:MeshData):
    from matplotlib import pyplot as plt
    
    plt.rcParams.update({
    "text.usetex": True,           # Use LaTeX for all text
    "font.family": "serif",        # Or 'sans-serif'
    "font.serif": ["Computer Modern"],   # LaTeX default
    "axes.unicode_minus": False,
    })
    
    
    def modify_ax(ax,fg):
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(direction='in', which='both', length=6, width=1, colors='black', grid_color='black', grid_alpha=0.5)
        ax.tick_params(axis='x', direction='in', which='both', bottom=False, top=True)
        ax.tick_params(axis='y', direction='in', which='both', left=True, right=False)
        #ax = arrowed_spines(fg, ax)
        
        return ax 
    
    fname = os.path.join(path2save, 'Slab_surface_comparison')
    if not os.path.isdir(fname):
        os.makedirs(fname)
    figure_name = f'Figure_{ipic:03d}.png'
    
    # Select the indices corresponding to the slab surface and oceanic crust
    z = M_data.X[:,1]
    
    ind_top  = (M_data.mesh_tag==8.0)| (M_data.mesh_tag==9.0)
    
    ind_ocr  = (M_data.mesh_tag==10.0)

    Ts_slab  = Ts[ind_top]
    
    Ts_ocmoh = Ts[ind_ocr]
    
    Ttd_slab = Ttd[ind_top]
    
    Ttd_ocmoh= Ttd[ind_ocr]

    z_s = z[ind_top]
    
    z_ocmoh = z[ind_ocr]
    
    ts = np.argsort(z_s)
    
    z_s = z_s[ts]
    
    Ts_slab = Ts_slab[ts]
    
    Ttd_slab = Ttd_slab[ts]
    
    tocmoh = np.argsort(z_ocmoh)
    
    z_ocmoh = z_ocmoh[tocmoh]
    
    Ts_ocmoh = Ts_ocmoh[tocmoh]
    
    Ttd_ocmoh = Ttd_ocmoh[tocmoh]
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(10,6))
    
    bx = 0.08 
    
    by = 0.04
    
    dy = 0.08 
    
    dx = 0.01 
    
    sx = 0.40
    
    sy = 0.4 
    
    ax0 = fig.add_axes([bx, by+sy+dy, sx, sy])
    
    ax1 = fig.add_axes([bx+sx+dx, by+sy+dy, sx, sy])
    
    ax2 = fig.add_axes([bx, by, sx, sy])
    
    ax3 = fig.add_axes([bx+sx+dx, by, sx, sy])
    
    ax0.plot(Ts_slab, z_s,c='forestgreen',label=r'$T^{\infty}$ Slab'                    ,linewidth=1.0, linestyle='-.')
    
    ax0.plot(Ttd_slab, z_s,c='firebrick',label=r'$T(t)$ Slab'     ,linewidth=1.2                )
    
    ax1.plot(Ts_ocmoh, z_ocmoh,c='forestgreen',label=r'$T^{\infty}$ Moho',linewidth=1.0,linestyle='-.'                )
    
    ax1.plot(Ttd_ocmoh, z_ocmoh,c='firebrick',label=r'$T(t)$ Moho',        linewidth=1.2)
    
    ax0.set_xlabel(r'T [$^{\circ} C$]', fontsize=12)
    
    ax0.set_ylabel('Depth [km]', fontsize=12)
    
    ax1.set_yticklabels([])
    
    ax3.set_yticklabels([])
    
    ax1.set_xlabel(r'T [$^{\circ} C$]', fontsize=12)
    
    ax2.plot(Ts_slab - Ttd_slab, z_s,c='darkblue',linewidth=1.2)
    
    ax2.set_xlabel(r'T$^{\infty}$ - T$(t)$ [$^{\circ}C$]', fontsize=12)

    ax2.set_ylabel('Depth [km]', fontsize=12)
    
    ax3.plot(Ts_ocmoh - Ttd_ocmoh, z_ocmoh,c='darkblue',linewidth=1.2)
        
    ax3.set_xlabel(r'T$^{\infty}$ - T$(t)$ [$^{\circ}C$]', fontsize=12)
    
    ax0 = modify_ax(ax0,fig)
    
    ax1 = modify_ax(ax1,fig)
    
    ax2 = modify_ax(ax2,fig)
    
    ax3 = modify_ax(ax3,fig)
    
    ax0.legend(fontsize=8, loc='lower left', frameon=False)
    ax1.legend(fontsize=8, loc='lower left', frameon=False)

    
    ax0.xaxis.set_label_position("top")
    
    ax1.xaxis.set_label_position("top")
    
    ax2.xaxis.set_label_position("top")
    
    ax3.xaxis.set_label_position("top")
    
    ax0.grid(True, linestyle='-.', linewidth=0.2)
    
    ax1.grid(True, linestyle='-.', linewidth=0.2)

    ax2.grid(True, linestyle='-.', linewidth=0.2)
    
    ax3.grid(True, linestyle='-.', linewidth=0.2)
    
    
    
    
    props = dict(boxstyle='round', facecolor='black', alpha=0.5)
    
    
    ax0.text(0.9, 1.15, '[a]', transform=ax0.transAxes, fontsize=8,
        verticalalignment='top', bbox=props,color='white')
    ax1.text(0.9, 1.15,'[b]', transform=ax1.transAxes, fontsize=8,
        verticalalignment='top', bbox=props,color='white')
    ax2.text(0.9, 1.15, '[c]', transform=ax2.transAxes, fontsize=8,
        verticalalignment='top', bbox=props,color='white')
    ax3.text(0.9, 1.15, '[d]', transform=ax3.transAxes, fontsize=8,
        verticalalignment='top', bbox=props,color='white')
 
    ax0.text(0.1, 1.15, time_string, transform=ax0.transAxes, fontsize=10,
        verticalalignment='top', bbox=props,color='white')
 
    fig.savefig(os.path.join(fname, figure_name))

    return 0
         
        
                
        

def create_figure(path2save:str, 
                  time_string, 
                  vmin:float, 
                  vmax:float, 
                  cmap: str, 
                  title: str,
                  M_data:MeshData,
                  field:float, 
                  n_level:int,
                  name_fig:str,
                  ipic:int,
                  name_field:str): 
    
    import matplotlib.pyplot as plt

    plt.rcParams.update({
    "text.usetex": True,           # Use LaTeX for all text
    "font.family": "serif",        # Or 'sans-serif'
    "font.serif": ["Computer Modern"],   # LaTeX default
    "axes.unicode_minus": False,
    })
    def modify_ax(ax):
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
        return ax

    xs = M_data.X[(M_data.mesh_tag==8.0) | (M_data.mesh_tag == 9.0),0]
    ys = M_data.X[(M_data.mesh_tag==8.0) | (M_data.mesh_tag == 9.0),1]
    sort = np.argsort(xs)
    
    x_s  = [xs[sort], ys[sort]]
    
    xor = M_data.X[(M_data.mesh_tag==11.0),0]
    yor = M_data.X[(M_data.mesh_tag==11.0),1]
    sort = np.argsort(xor)
    
    x_or  = [xor[sort], yor[sort]]
    
    xbt = M_data.X[(M_data.mesh_tag==6.0),0]
    ybt = M_data.X[(M_data.mesh_tag==6.0),1]
    sort = np.argsort(xbt)
    
    x_bt  = [xbt[sort], ybt[sort]]
    
    
    
    pt_save = os.path.join(path2save,name_fig)
    if not os.path.isdir(pt_save):
        os.makedirs(pt_save)
        
    figure_name = f'Figure_{ipic:03d}_%s.png'%name_field
        
    fname = os.path.join(pt_save, figure_name)
    fig, ax0 = plt.subplots(figsize=(10, 6))
    ax0 = modify_ax(ax0)
    ax0.set_title(time_string, fontsize=16)
    ax0.set_xlabel('Distance [km]', fontsize=14)
    ax0.set_ylabel('Depth [km]', fontsize=14)
    p0 = ax0.contourf(M_data.Xi, M_data.Yi, field, levels=n_level, cmap=cmap, vmin=vmin, vmax=vmax)
    p1 = ax0.plot(x_s[0],x_s[1],c='w',linewidth=1.2)
    p2 = ax0.plot(x_or[0],x_or[1],c='w',linewidth=1.2)
    p3 = ax0.plot(x_bt[0],x_bt[1],c='k',linewidth=1.2)
    cbar = plt.colorbar(p0, ax=ax0, orientation='vertical', pad=0.02)
    cbar.set_label(label=title, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    #cbar.set_ticks(np.linspace(vmin, vmax, 5))
    #cbar.set_ticklabels([f'{val:.0f}' for val in np.linspace(vmin, vmax, 5)])
    fig.savefig(fname)
    plt.close(fig)
    
    return 0 


def return_main_boundarys(M_data:MeshData,tag:float,sort_axis:int):
    
    x = M_data.X[(M_data.mesh_tag==tag),0]
    y = M_data.X[(M_data.mesh_tag==tag),1]
    
    if sort_axis == 0:
        sort = np.argsort(x)
    else:
        sort = np.argsort(y)
    
    xbd  = [x[sort], y[sort]]
    
    
    return xbd

def plot_main_boundarys(path_2_save,time_str,ipic,M_data): 
    
    import matplotlib.pyplot as plt

    plt.rcParams.update({
    "text.usetex": True,           # Use LaTeX for all text
    "font.family": "serif",        # Or 'sans-serif'
    "font.serif": ["Computer Modern"],   # LaTeX default
    "axes.unicode_minus": False,
    })
    def modify_ax(ax):
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
        return ax

    def _plot_(x_bd,ax,c,ls,lw=1.2,opt=0):
        if opt ==0:
            p = ax.plot(x_bd[0],x_bd[1],c=c,linewidth=1.2)
            return p
        else: 
            pa = ax.plot(x_bd[0][x_bd[1]>-80],x_bd[1][x_bd[1]>-80],c='orange',linewidth=lw,linestyle=ls)
            pb = ax.plot(x_bd[0][x_bd[1]<=-80],x_bd[1][x_bd[1]<=-80],c='firebrick',linewidth=lw,linestyle=ls)
            return pa,pb
    
    
    name_fig = 'Main_boundaries'



    bound = lambda tag, axis: return_main_boundarys(M_data,tag,axis)


    pt_save = os.path.join(path_2_save,name_fig)
    if not os.path.isdir(pt_save):
        os.makedirs(pt_save)
        
    figure_name = f'Boundary.png'
        
    fname = os.path.join(pt_save, figure_name)
    fig, ax0 = plt.subplots(figsize=(10, 6))
    ax0 = modify_ax(ax0)
    ax0.set_xlabel('Distance [km]', fontsize=14)
    ax0.set_ylabel('Depth [km]', fontsize=14)
    p1 = _plot_(bound(8.0,0),ax0,'orange','-')   # Slab surface
    p2,p3 = _plot_(bound(9.0,0),ax0,'w','-',opt=1)   # Slab surface
    p4 = _plot_(bound(11.0,0),ax0,'k','-') 
    p5 = _plot_(bound(1.0,0),ax0,'k','-')  
    p6 = _plot_(bound(2.0,1),ax0,'k','-')  
    p7 = _plot_(bound(3.0,1),ax0,'forestgreen','..-')  
    p8 = _plot_(bound(4.0,0),ax0,'forestgreen','..-')  
    p9 = _plot_(bound(5.0,0),ax0,'forestgreen','..-')  
    p10 = _plot_(bound(6.0,0),ax0,'k','-')  
    p11 = _plot_(bound(7.0,1),ax0,'forestgreen','..-')  
    #p12 = _plot_(bound(12.0,1),ax0,'grey','..-',lw=0.3)  
    #p13 = _plot_(bound(13.0,1),ax0,'grey','..-',lw=0.3)  
    #p14 = _plot_(bound(10.0,1),ax0,'grey','..-',lw=0.3)  
    
    fig.savefig(fname)
    plt.close(fig)
    
    
    
    #----- Create Three poligons for masking -----#
    x_slab = bound(8.0,0) 
    x_slab2 = bound(9.0,0) 
    x_bottom = bound(5.0,0)
    x_slab_bottom = bound(6.0,0)
    x_left = bound(7.0,0)
    
    from matplotlib.patches import Polygon        
    from shapely import contains_xy as scontains_xy
        
    
    fname = os.path.join(pt_save, figure_name)
    fig, ax0 = plt.subplots(figsize=(10, 6))
    polygon = np.vstack((np.array(x_left).T,
                         np.array(x_slab_bottom).T, 
                          np.array(x_bottom).T,
                          np.array(x_slab2).T, 
                        np.array(x_slab).T))

    return 0 
def compare_SS_TD(ss_file:str, td_file:str, time_td, M_data:MeshData,path_2_save:str):

    
    """
    Easy peasy function to compare steady state and time dependent solution at given time. Moreover, to make nice plot rather than 
    that crap from paraview. 
    1st: load the timedependent file and extract the geometry and temperature at given time
         1. find the outerboundaries of the domain (very ugly way)
         2. create a polygon from these boundaries
         3. create a grid and interpolate the temperature on it
         4. mask the temperature outside the polygon
         
    2nd: load the steady state file and extract the temperature
         1. find the outerboundaries of the domain (very ugly way)
         2. create a polygon from these boundaries
         3. create a grid and interpolate the temperature on it
         4. mask the temperature outside the polygon
    3rd: make the plot
    4th: save the plot
    """
    
    
    f = h5py.File(td_file, 'r')
    field = 'Function/Temperature  [degC]'
    times = list(f[field].keys())
    time_list = [float(s.replace("_", ".")) for s in times]
    time_sort = np.argsort(time_list)
    time_list = [time_list[i] for i in time_sort]
    times = [times[i] for i in time_sort]
    
    fs = h5py.File(ss_file, 'r')
    field = 'Function/Temperature  [degC]'
    TS = np.array(fs[field+'/0'])
    
    


    # Load steady state file    
    T_S = interpolate_field(TS,M_data)

    
    ipic     = 0 

    # CREATE FIGURE FOR STEADY STATE
    time_str = r'T = $\infty$'
    
    create_figure(path_2_save,time_str,20,1300,'cmc.lipari',r'Temperature [$^{\circ}C$]', M_data,T_S, 20, 'TSS',ipic,)

    # Extract Steady state heat fluxes
    qS = np.array(fs['Function/q  [W/m2]/0'])
    
    qSx = qS[:,0]
    
    qSy = qS[:,1]
    
    qSm = np.sqrt(qSx**2 + qSy**2)
    
    qsxi = interpolate_field(qSx,M_data)
    
    qsyi = interpolate_field(qSy,M_data)
    
    qsmi = interpolate_field(qSm,M_data)
    
    
    
    
    for i in times:
        t = time_list[ipic]
        
        time_str = f'Time = {t:.3f} Myr'
        
        T = np.array(f[field+'/'+i])
        
        dT = TS-T 
        
        vmin_dt = np.floor(np.min(dT))
        
        vmax_dt = np.floor(np.max(dT))
        
        dT_i = interpolate_field(dT,M_data)
        
        T_i  = interpolate_field(T,M_data)

        create_figure(path_2_save,time_str,20,1300,'cmc.lipari',r'Temperature [$^{\circ}C$]', M_data,T_i, 20, 'T',ipic)
        
        create_figure(path_2_save,time_str,vmin_dt,vmax_dt,'cmc.vik',r'T$^{\infty}$-T$(t)$ [$^{\circ}C$]', M_data,dT_i, 20, 'dT',ipic)
        
        compare_slab_surface(path_2_save,
                            time_str,
                            ipic,
                            TS,
                            T,
                            M_data)    
        
        qtd = np.array(f['Function/q  [W/m2]/'+i])
        qTdx = qtd[:,0]
        qTdy = qtd[:,1]
        qTdm = np.sqrt(qTdx**2 + qTdy**2)  
        compare_heatfluxes(path_2_save,
                           time_str,
                           ipic,
                           qSx,
                           qSy,
                           qSm,
                           qTdx,
                           qTdy,
                           qTdm,
                           M_data)
        
        ipic = ipic + 1 
        
    
    
    
    
    
    
    return 0 


def compare_experiments(file1:str, file2:str, time_td, M_data:MeshData,path_2_save:str,name_exp:list):

    
    """
    Easy peasy function to compare steady state and time dependent solution at given time. Moreover, to make nice plot rather than 
    that crap from paraview. 
    1st: load the timedependent file and extract the geometry and temperature at given time
         1. find the outerboundaries of the domain (very ugly way)
         2. create a polygon from these boundaries
         3. create a grid and interpolate the temperature on it
         4. mask the temperature outside the polygon
         
    2nd: load the steady state file and extract the temperature
         1. find the outerboundaries of the domain (very ugly way)
         2. create a polygon from these boundaries
         3. create a grid and interpolate the temperature on it
         4. mask the temperature outside the polygon
    3rd: make the plot
    4th: save the plot
    """
    
    
    f = h5py.File(file1, 'r')
    field = 'Function/Temperature  [degC]'
    TSa = np.array(f[field+'/0'])
    fs = h5py.File(file2, 'r')
    TSb = np.array(fs[field+'/0'])
    dTS = TSa - TSb
    


    # Load steady state file    
    T_S = interpolate_field(dTS,M_data)

    
    ipic     = 0 

    # CREATE FIGURE FOR STEADY STATE
    time_str = r'time = $\infty$'
    lim = [np.floor(np.nanmin(dTS)),np.floor(np.nanmax((dTS)))]
    
    create_figure(path_2_save,time_str,lim[0],lim[1],'cmc.vik',r'$\Delta$ T (%s-%s) [$^{\circ}C$]'%(name_exp[0],name_exp[1]), M_data,T_S, 20, 'TSS',ipic,'dT')


    compare_slab_surface_SS(path_2_save,
                            time_str,
                            ipic,
                            TSa,
                            TSb,
                            M_data,
                            name_exp)    
    
    
    
    
    return 0 


def plot_ss_temperature(file1:str, M_data:MeshData,path_2_save:str,name_exp:list):

    
    """
    Easy peasy function to compare steady state and time dependent solution at given time. Moreover, to make nice plot rather than 
    that crap from paraview. 
    1st: load the timedependent file and extract the geometry and temperature at given time
         1. find the outerboundaries of the domain (very ugly way)
         2. create a polygon from these boundaries
         3. create a grid and interpolate the temperature on it
         4. mask the temperature outside the polygon
         
    2nd: load the steady state file and extract the temperature
         1. find the outerboundaries of the domain (very ugly way)
         2. create a polygon from these boundaries
         3. create a grid and interpolate the temperature on it
         4. mask the temperature outside the polygon
    3rd: make the plot
    4th: save the plot
    """
    
    
    f = h5py.File(file1, 'r')
    field = 'Function/Temperature  [degC]'
    TSS = np.array(f[field+'/0'])
    k   = 'Function/k  [W/m/k]/0'
    k   = np.array(f[k]) 
    k = interpolate_field(k,M_data) 


    # Load steady state file    
    T_S = interpolate_field(TSS,M_data)

    
    ipic     = 0 

    # CREATE FIGURE FOR STEADY STATE
    time_str = r'time = $\infty$'
    lim = [0,2000]
    lims = [np.floor(np.nanmin(k)),np.floor(np.nanmax(k))]
    
    
    plot_main_boundarys(path_2_save,time_str,ipic,M_data)

    
    
    create_figure(path_2_save,time_str,lim[0],lim[1],'cmc.lipari',r'T [$^{\circ}C$]', M_data,T_S, 21, 'TSS',ipic,'T')

    create_figure(path_2_save,time_str,lims[0],lims[1],'cmc.vik',r'k [W/m/K]', M_data,k, 40, 'kSS',ipic,'k')
    
    
    
    
    return 0 


if __name__ == "__main__":
    path_2_test = '/Users/wlnw570/Work/Results/T0/Output'
    path_2_testb = '/Users/wlnw570/Work/Results/T0b/Output'

    path_2_save = '/Users/wlnw570/Work/Output_deb/T0'
    path_2_saveb = '/Users/wlnw570/Work/Output_deb/Comparison_T0_T0b'
    if not os.path.isdir(path_2_save):
        os.makedirs(path_2_save)
    
    td_file = '%s/time_dependent.h5'%(path_2_test)
    ss_file = '%s/Steady_state.h5'%(path_2_test)
    ss_file2 = '%s/Steady_state.h5'%(path_2_testb)
    td_file2 = '%s/time_dependent.h5'%(path_2_testb)



    ms_tag  = '%s/MeshTag.h5'%(path_2_test)
    time_td = 10.0  # Time in Myr to compare
    M_data = MeshData(ss_file,ms_tag)
    
    plot_ss_temperature(ss_file, M_data, path_2_save,'T0')
    
    compare_experiments(ss_file, ss_file2, time_td, M_data, path_2_saveb,['T0','T0b'])
    
    
    #compare_SS_TD(ss_file2, td_file2, time_td, M_data, path_2_save)
    '''