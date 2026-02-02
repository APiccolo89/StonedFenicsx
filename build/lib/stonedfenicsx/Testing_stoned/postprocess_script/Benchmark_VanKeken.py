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
    
    sx = 0.38
    
    sy = 0.40
    
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
                  field:list, 
                  n_level:int,
                  name_fig:str,
                  ipic:int,
                  name_field:str,
                  labelcb:str): 
    
    import matplotlib.pyplot as plt

    plt.rcParams.update({
    "text.usetex": True,           # Use LaTeX for all text
    "font.family": "serif",        # Or 'sans-serif'
    "font.serif": ["Computer Modern"],   # LaTeX default
    "axes.unicode_minus": False,
    })
    
    def define_colorbar_S(cf,ax,lim:list,ticks:list,label:str):
        
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
        cbaxes = inset_axes(ax, borderpad=1.3,  width="100%", height="15%", loc=3)   
        cbar=plt.colorbar(cf,cax=cbaxes, ticks=ticks, orientation='horizontal',extend="both")
        print(lim[0])
        print(lim[1])
        cbar.set_label(label=label,size=10) 
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_coords(0.5,-0.15)
        cbar.ax.xaxis.set_tick_params(pad=0.1)
        cbar.ax.xaxis.set_label_position('bottom')
        cbar.ax.tick_params(labelsize=12)
    
        return cbaxes,cbar
    # Prepare axis 
    def modify_ax(ax):
        ax.spines['bottom'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['bottom'].set_color('black')
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
    bx = 0.07
    by = 0.20
    dy = 0.01
    dx = 0.01
    sx = 0.45    
    sy = 0.25
    
    
    
    
    fig = plt.Figure(figsize=(18, 12))
    
    
    ax0 = fig.add_axes([bx, by, sx, sy])
    ax1 = fig.add_axes([bx, by+dy+sy, sx, sy])
    ax2 = fig.add_axes([bx, by+2*(dy+sy), sx, sy])
    ax3 = fig.add_axes([bx+dx+sx, by, sx, sy])
    ax4 = fig.add_axes([bx+dx+sx, by+dy+sy, sx, sy])
    ax5 = fig.add_axes([bx+dx+sx, by+2*(dy+sy), sx, sy])
    #ax6 = fig.add_axes([0.25, 0.01, 0.5, 0.14])
    ax7 = fig.add_axes([0.25, 0.01+0.07+0.01, 0.5, 0.07])



    vdTick = vmin
    VdTick = vmax

        
    ax0 = modify_ax(ax0) # dT Reference -> adiabatic 
    ax0.set_title(time_string, fontsize=16)
    ax0.set_xlabel('Distance [km]', fontsize=14)
    ax0.set_ylabel('Depth [km]', fontsize=14)
    p0 = ax0.pcolormesh(M_data.Xi, M_data.Yi, field[0],  cmap=cmap, vmin=vdTick, vmax=VdTick)
    p1 = ax0.plot(x_s[0],x_s[1],c='w',linewidth=1.2)
    p2 = ax0.plot(x_or[0],x_or[1],c='w',linewidth=1.2)
    p3 = ax0.plot(x_bt[0],x_bt[1],c='k',linewidth=1.2)
    
    #cbar.set_ticks(np.linspace(vmin, vmax, 5))
    #cbar.set_ticklabels([f'{val:.0f}' for val in np.linspace(vmin, vmax, 5)])
    field_1 = field[1]
    ax1 = modify_ax(ax1) # T Adiabatic 
    ax1.set_ylabel('Depth [km]', fontsize=14)
    ax1.set_xticklabels([])
    ax1.set_xlabel('', fontsize=14)
    
    p0 = ax1.pcolormesh(M_data.Xi, M_data.Yi, field[1], cmap=cmap, vmin=vdTick, vmax=VdTick)
    p1 = ax1.plot(x_s[0],x_s[1],c='w',linewidth=1.2)
    p2 = ax1.plot(x_or[0],x_or[1],c='w',linewidth=1.2)
    p3 = ax1.plot(x_bt[0],x_bt[1],c='k',linewidth=1.2)
    
    field_1 = field[0]
    ax2 = modify_ax(ax2) # T without adiabatic 
    ax2.set_ylabel('Depth [km]', fontsize=14)
    ax2.set_xticklabels([])
    pB = ax2.pcolormesh(M_data.Xi, M_data.Yi, field[2], cmap=cmap, vmin=vdTick, vmax=VdTick)
    p1 = ax2.plot(x_s[0],x_s[1],c='w',linewidth=1.2)
    p2 = ax2.plot(x_or[0],x_or[1],c='w',linewidth=1.2)
    p3 = ax2.plot(x_bt[0],x_bt[1],c='k',linewidth=1.2)
    
    ax3 = modify_ax(ax3) # dT adiabatic reference,  
    pA = ax3.pcolormesh(M_data.Xi, M_data.Yi, field[3],  cmap=cmap,  vmin=vdTick, vmax=VdTick)
    ax3.set_xlabel('Distance [km]', fontsize=14)
    
    ax3.set_yticklabels([])
    p1 = ax3.plot(x_s[0],x_s[1],c='w',linewidth=1.2)
    p2 = ax3.plot(x_or[0],x_or[1],c='w',linewidth=1.2)
    p3 = ax3.plot(x_bt[0],x_bt[1],c='k',linewidth=1.2)
    
    ax4 = modify_ax(ax4) # dT adiabatic reference,  
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    pB = ax4.pcolormesh(M_data.Xi, M_data.Yi, field[4], cmap=cmap,  vmin=vdTick, vmax=VdTick)
    p1 = ax4.plot(x_s[0],x_s[1],c='w',linewidth=1.2)
    p2 = ax4.plot(x_or[0],x_or[1],c='w',linewidth=1.2)
    p3 = ax4.plot(x_bt[0],x_bt[1],c='k',linewidth=1.2)
    
    ax5 = modify_ax(ax5) # dT adiabatic reference,  
    ax5.set_xticklabels([])
    ax5.set_yticklabels([])
    pT = ax5.pcolormesh(M_data.Xi, M_data.Yi, field[5],  cmap=cmap,  vmin=vdTick, vmax=VdTick)
    p1 = ax5.plot(x_s[0],x_s[1],c='w',linewidth=1.2)
    p2 = ax5.plot(x_or[0],x_or[1],c='w',linewidth=1.2)
    p3 = ax5.plot(x_bt[0],x_bt[1],c='k',linewidth=1.2)
    
    cb0 = define_colorbar_S(pB,ax7,[vdTick,VdTick],np.linspace(vdTick,VdTick,10),label=labelcb)
    ax7.set_axis_off()

    props = dict(boxstyle='round', facecolor='black', alpha=0.5)

    
    ax0.text(0.01, 0.1, '[c]', transform=ax0.transAxes, fontsize=12,
        verticalalignment='top', bbox=props,color='white')
    ax1.text(0.01, 0.1,'[b]', transform=ax1.transAxes, fontsize=12,
        verticalalignment='top', bbox=props,color='white')
    ax2.text(0.01, 0.1, '[a]', transform=ax2.transAxes, fontsize=12,
        verticalalignment='top', bbox=props,color='white')
    ax3.text(0.01, 0.1, '[f]', transform=ax3.transAxes, fontsize=12,
        verticalalignment='top', bbox=props,color='white')
    ax4.text(0.01, 0.1, r'[e]', transform=ax4.transAxes, fontsize=12,
        verticalalignment='top', bbox=props,color='white')
    ax5.text(0.01, 0.1, '[d]', transform=ax5.transAxes, fontsize=12,
        verticalalignment='top', bbox=props,color='white')  
 
    
    fig.savefig(fname)
    plt.close(fig)
    
    return 0 

def _benchmark_van_keken(T,XG,case):
    from scipy.interpolate import griddata
    


    # gather solution from all processes on proc 0
    
 

    x_g = np.array([np.min(XG[:,0]),np.max(XG[:,0])],dtype=np.float64)
    y_g = np.array([np.min(XG[:,1]),np.max(XG[:,1])],dtype=np.float64)
    nx   = 111 
    ny   = 101
    idx0 = 10 #(11 in van keken)
    idy0 = ny-11 
    idx1 = 36-1 
    idy1 = ny-36

    xx   = np.linspace(x_g[0],x_g[1],num=nx)
    yy   = np.linspace(y_g[0],y_g[1],num=ny)
    T_g  = np.zeros([nx,ny],dtype=np.float64)
    X,Y  = np.meshgrid(xx,yy)
    xt,yt = XG[:,0], XG[:,1]
    p     = np.zeros((len(xt),2),dtype=np.float64)
    p[:,0] = xt 
    p[:,1] = yt
    T_g   = griddata(p,T,(X, Y), method='nearest')
    T_g = T_g.transpose()
    T = 0
    co = 0
    x_s=[]
    y_s=[] 
    T2 = []
    c = 0
    X_S = []
    Y_S = []
    i_X = 10
    for i in range(36):
            T = T+(T_g[i,ny-1-i])**2
            x_s.append(xx[i])
            y_s.append(yy[ny-1-i])
            if (i<21) & (i>8):
                l_index = np.arange(ny-(i+1),ny-10)
                if len(l_index) == 0:
                    l_index_1st = np.arange(9,21)
                    for j in range(len(l_index_1st)):
                        T2.append(T_g[l_index_1st[j],ny-(i+1)]**2)
                        X_S.append(xx[l_index_1st[j]])
                        Y_S.append(yy[ny-(i+1)])
                        c = c + 1
                for j in range(len(l_index)):
                    T2.append(T_g[i,l_index[j]]**2)
                    X_S.append(xx[i])
                    Y_S.append(yy[int(l_index[j])])
                    c = c + 1

            co = co+1 

    T_11_11 = T_g[idx0,idy0]

    T_ln = np.sqrt(T/co) 
    T_ln2 = np.sqrt(np.sum(T2)/c)

    data_1c = np.array([
    [397.55, 505.70, 850.50],
    [391.57, 511.09, 852.43],
    [387.78, 503.10, 852.97],
    [387.84, 503.13, 852.92],
    [389.39, 503.04, 851.68],
    [388.73, 504.03, 854.99]
    ])

    data_2a = np.array([
    [570.30, 614.09, 1007.31],
    [580.52, 606.94, 1002.85],
    [580.66, 607.11, 1003.20],
    [577.59, 607.52, 1002.15],
    [581.30, 607.26, 1003.35]
    ])

    data_2b = np.array([
    [550.17, 593.48, 994.11],
    [551.60, 608.85, 984.08],
    [582.65, 604.51, 998.71],
    [583.36, 605.11, 1000.01],
    [574.84, 603.80, 995.24],
    [583.11, 604.96, 1000.05]
    ])

    if case    == 0: 
        data   = data_1c 
    elif case  == 1:
        data = data_2a
    elif case== 2 : 
        data = data_2b 
    else: 
        data = data_1c*0.

    print( '------------------------------------------------------------------' )
    print( ':::====> T(11,11) = %.2f [deg C], average value VanK2008 = %.2f [deg C] +/- %.2f;m/M = %.2f/%.2f [deg C]'%( T_11_11, np.mean(data[:,0]), np.std(data[:,0]),np.min(data[:,0]),np.max(data[:,0]) ) )
    print( ':::====> L2_A     = %.2f [deg C], average value VanK2008 = %.2f [deg C] +/- %.2f;m/M = %.2f/%.2f [deg C]'%( T_ln, np.mean(data[:,1]), np.std(data[:,1]) ,np.min(data[:,1]),np.max(data[:,1]) ) )
    print( ':::====> L2_B     = %.2f [deg C], average value VanK2008 = %.2f [deg C] +/- %.2f;;m/M = %.2f/%.2f[deg C]'%( T_ln2, np.mean(data[:,2]), np.std(data[:,2]),np.min(data[:,2]),np.max(data[:,2]) ) )
    print( ':::L2_A = T along the slab surface from 0 to -210 km' )
    print( ':::L2_B = T wedge norm from -54 to -110 km ' )
    print( ':::=> From grid to downsampled grid -> nearest interpolation' )
    print( '------------------------------------------------------------------' )
    # place holder for the save database 
        


    return 0

path = '/Users/wlnw570/Work/Leeds/Results/Tests_Van_keken'
path2save = '/Users/wlnw570/Work/Leeds/Results/Figures'
if not os.path.isdir(path2save):
    os.makedirs(path2save)  
    

T0 = Test('%s/Test_VK_option_0'%path)
T1 = Test('%s/Test_VK_option_1'%path)
T2 = Test('%s/Test_VK_option_2'%path)
T3 = Test('%s/Test_VK_option_NL_0'%path)
T4 = Test('%s/Test_VK_option_NL_1'%path)
T5 = Test('%s/Test_VK_option_NL_2'%path)


T6 = Test('%s/Test_VK_option_AD_0'%path)
T7 = Test('%s/Test_VK_option_AD_1'%path)
T8 = Test('%s/Test_VK_option_AD_2'%path)
T9 = Test('%s/Test_VK_option_NL_AD_0'%path)
T10 = Test('%s/Test_VK_option_NL_AD_1'%path)
T11 = Test('%s/Test_VK_option_NL_AD_2'%path)

T12 = Test('%s/Test_VK_option_AD_SC_0'%path)
T13 = Test('%s/Test_VK_option_AD_SC_1'%path)
T14 = Test('%s/Test_VK_option_AD_SC_2'%path)

Temp_T0 = T0._interpolate_data('SteadyState.Temp')
Temp_T1 = T1._interpolate_data('SteadyState.Temp')
Temp_T2 = T2._interpolate_data('SteadyState.Temp')
Temp_T3 = T3._interpolate_data('SteadyState.Temp')
Temp_T4 = T4._interpolate_data('SteadyState.Temp')
Temp_T5 = T5._interpolate_data('SteadyState.Temp')


Temp_T6 = T6._interpolate_data('SteadyState.Temp')
Temp_T7 = T7._interpolate_data('SteadyState.Temp')
Temp_T8 = T8._interpolate_data('SteadyState.Temp')
Temp_T9 = T9._interpolate_data('SteadyState.Temp')
Temp_T10 = T10._interpolate_data('SteadyState.Temp')
Temp_T11 = T11._interpolate_data('SteadyState.Temp')

Temp_T12 = T12._interpolate_data('SteadyState.Temp')
Temp_T13 = T13._interpolate_data('SteadyState.Temp')
Temp_T14 = T14._interpolate_data('SteadyState.Temp')

kappa0 = T0._interpolate_data('SteadyState.kappa')
kappa1 = T1._interpolate_data('SteadyState.kappa')
kappa2 = T2._interpolate_data('SteadyState.kappa')
kappa3 = T3._interpolate_data('SteadyState.kappa')
kappa4 = T4._interpolate_data('SteadyState.kappa')
kappa5 = T5._interpolate_data('SteadyState.kappa')
kappa6 = T6._interpolate_data('SteadyState.kappa')
kappa7 = T7._interpolate_data('SteadyState.kappa')
kappa8 = T8._interpolate_data('SteadyState.kappa')
kappa9 = T9._interpolate_data('SteadyState.kappa')
kappa10 = T10._interpolate_data('SteadyState.kappa')
kappa11 = T11._interpolate_data('SteadyState.kappa')

fields = [Temp_T0,Temp_T1,Temp_T2,Temp_T3,Temp_T4,Temp_T5]
fieldsa = [Temp_T6,Temp_T7,Temp_T8,Temp_T9,Temp_T10,Temp_T11]


fields2 = [Temp_T3-Temp_T0,Temp_T4-Temp_T1,Temp_T5-Temp_T2,Temp_T9-Temp_T6,Temp_T10-Temp_T7,Temp_T11-Temp_T8]
fields3 = [Temp_T9-Temp_T6,Temp_T10-Temp_T7,Temp_T11-Temp_T8,Temp_T12-Temp_T6,Temp_T13-Temp_T7,Temp_T14-Temp_T8]



fields_kappa = np.log10([kappa3,kappa4,kappa5,kappa9,kappa10,kappa11])

M_data = T0.MeshData

vmin = 0.0
vmax = 1650.0 

n_level = 20
cmap = cmcrameri.cm.lipari
cmap2 = cmcrameri.cm.vik
cmap3 = cmcrameri.cm.bamako

title = 'Temperature [$^{\circ}C$]'
name_fig = 'Temperature'
time_string = 'Steady State'

create_figure(path2save,M_data,0.0,1300,cmap,title,fields,n_level,name_fig,0,'comparison',r'T [$^{\circ} C$]')
create_figure(path2save,M_data,0.0,1650,cmap,title,fieldsa,n_level,name_fig,0,'comparison_AD',r'T [$^{\circ} C$]')
create_figure(path2save,M_data,-150,150,cmap2,title,fields2,n_level,name_fig,0,'dT',r'$\Delta$ T [$^{\circ} C$]')
create_figure(path2save,M_data,-150,150,cmap2,title,fields3,n_level,name_fig,0,'dT2',r'$\Delta$ T [$^{\circ} C$]')
create_figure(path2save,M_data,-6.5,-5.5,cmap3,title,fields_kappa,10,name_fig,0,'kappa',r'$log_{10}(\kappa)$ [m$^2$/s]')

Xgl = T1.MeshData.X
_benchmark_van_keken(T0.Data_raw.SteadyState.Temp,Xgl,0)
_benchmark_van_keken(T1.Data_raw.SteadyState.Temp,Xgl,1)
_benchmark_van_keken(T2.Data_raw.SteadyState.Temp,Xgl,2)
