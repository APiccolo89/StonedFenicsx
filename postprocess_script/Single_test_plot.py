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
    

T5 = Test('%s/exp4_bisNS/Output'%path)


Temp_T5 = T5._interpolate_data('SteadyState.Temp')


rho = T5._interpolate_data('SteadyState.rho')


Cp = T5._interpolate_data('SteadyState.Cp')


alpha = T5._interpolate_data('SteadyState.alpha')

eta = T5._interpolate_data('SteadyState.eta')

k = T5._interpolate_data('SteadyState.k')




M_data = T5.MeshData

vmin = 0.0
vmax = 1700.0 

n_level = 100
cmap = cmcrameri.cm.lipari
title = 'Temperature [$^{\circ}C$]'
name_fig = 'Temperature'
time_string = 'Steady State'

create_figure(path2save,M_data,vmin,np.nanmax(Temp_T5),cmap,title,Temp_T5,n_level,name_fig,0,'e_4_YA_NL_C_RB')   


vmin = 0.0
vmax = 1700.0 

n_level = 100
cmap = cmcrameri.cm.vik
title = r'$\frac{\rho}{\rho^{M}_0}$ []'
name_fig = 'Density'
time_string = 'Steady State'

create_figure(path2save,M_data,np.nanmin(rho/3330),np.nanmax(rho/3330),cmap,title,rho/3330,n_level,name_fig,0,'e_4_YA_NL_C_RB') 


vmin = 0.0
vmax = 1700.0 

n_level = 10
cmap = cmcrameri.cm.berlin
title = r'$\frac{Cp}{Cp^{M}_0}$ []'
name_fig = 'Specific_heat'
time_string = 'Steady State'

create_figure(path2save,M_data,np.nanmin(Cp/1250),np.nanmax(Cp/1250),cmap,title,Cp/1250,n_level,name_fig,0,'e_4_YA_NL_C_RB') 


vmin = 0.0
vmax = 1700.0 

n_level = 10
cmap = cmcrameri.cm.bam
title = r'$\frac{k}{k^{M}_0}$ []'
name_fig = 'Heat_conductivity'
time_string = 'Steady State'

create_figure(path2save,M_data,np.nanmin(k/3.0),np.nanmax(k/3.0),cmap,title,k/3.0,n_level,name_fig,0,'e_4_YA_NL_C_RB') 


vmin = 0.0
vmax = 1700.0 

n_level = 10
cmap = cmcrameri.cm.managua
title = r'$\frac{\eta}{\eta^{M}_0}$ []'
name_fig = 'Viscosity'
time_string = 'Steady State'

create_figure(path2save,M_data,np.log10(np.nanmin(eta/1e21)),np.log10(np.nanmax(eta/1e21)),cmap,title,np.log10(eta/1e21),n_level,name_fig,0,'e_4_YA_NL_C_RB') 

vmin = 0.0
vmax = 1700.0 

n_level = 10
cmap = cmcrameri.cm.vik
title = r'$\frac{\alpha}{\alpha^{M}_0}$ []'
name_fig = 'alpha'
time_string = 'Steady State'

create_figure(path2save,M_data,np.nanmin(alpha/3e-5),np.nanmax(alpha/3e-5),cmap,title,alpha/3e-5,n_level,name_fig,0,'e_4_YA_NL_C_RB') 
# --- 



