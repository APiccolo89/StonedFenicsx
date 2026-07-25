# -- 
import h5py
import numpy as np
import os
import numpy as np 
import matplotlib.pyplot as plt 
from data_extractor import Test 
import pathlib 
from pathlib import Path
from typing import NamedTuple
from numpy.typing import NDArray
import cmcrameri
# -- 
class boundary_mesh(NamedTuple):
    top_surface_slab : NDArray 
    bot_surface_slab : NDArray 
    top_model:NDArray 
    bot_model:NDArray 
    ns_bond:NDArray 
    right_bd:NDArray 
    left_bd : NDArray 
    
# --
# Script to visualise the case 2c of Van Keken suite and the case 2c with non-linearities (crust+mantle)
# 1. Extract the data for these two test -> From the tests s.s. and the benchmark database of van keken 
# 2. Prepare a 2D plot with temperature above [column A -> case 2c ][column B->2c+NL]
# 3. Prepare 4 other plot with the residual evolution 

# --
# Conversion pt - cm 
_PT_CM_ = 0.035
# Ticks, label and text font size 
_TICKSFONTSIZE_ = 0.8/_PT_CM_
_CBTICKSFONTSIZE_ = 0.6/_PT_CM_
_CBLABELFONTSIZE_ = 0.8/_PT_CM_
_LABELFONTSIZE_ = 1.0/_PT_CM_
_TEXTFONTSIZE_ = 0.65/_PT_CM_
TRANSPARENT = True 
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",   # Computer Modern
})

def plot_colorbar_and_deactivate(fig, cax, mappable, **kwargs):
    """
    Plot a colorbar on a pre-existing axis and deactivate the axis afterwards.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object.
    cax : matplotlib.axes.Axes
        Dedicated colorbar axis.
    mappable : matplotlib.cm.ScalarMappable
        Image/contour/pcolormesh object.
    kwargs :
        Extra arguments for fig.colorbar.

    Returns
    -------
    cb : matplotlib.colorbar.Colorbar
    """

    cb = fig.colorbar(mappable, cax=cax,**kwargs)
    cb.ax.tick_params(labelsize=_CBTICKSFONTSIZE_)
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.xaxis.set_label_position("bottom")
    cb.ax.tick_params(pad=3.0)  
    # Distance (points) between ticks and tick labels    
    # Hide only the frame, keep ticks and labels
    for spine in cax.spines.values():
        spine.set_visible(False)
    # move label to top
    cb.ax.xaxis.set_label_position('top')
    cb.set_label(
    r'T, $[^{\circ}C]$',
    fontsize=_CBLABELFONTSIZE_,labelpad=3.0)    
    return cb
# -- 
def extract_boundary_mesh(test:Test)->boundary_mesh:
    from stonedfenicsx.create_mesh.aux_create_mesh import dict_tag_lines
    # closure for finding x,z mesh
    def find_x_y_reorder(boundary:int|list,sorting_direction='x')->NDArray:
        # Choose the mesh tag
        if type(boundary) is list: 
            chosen = (mesh_tag == boundary[0]) | (mesh_tag == boundary[1])
        else: 
            chosen = mesh_tag == boundary
            
            
        # Choose the coordinates
        x = test.MeshData.X[chosen,0]
        y = test.MeshData.X[chosen,1]
        # sort the data 
        if sorting_direction == 'x': 
            ind_sort = np.argsort(x)
        else:
            ind_sort = np.argsort(y)
        
        x = x[ind_sort]
        y = y[ind_sort]
        
        bnd = np.zeros([len(x),2],dtype=float)
        bnd[:,0] = x
        bnd[:,1] = y 
        return bnd 
    
    # extract_top_surface_slab 
    mesh_tag = test.MeshData.mesh_tag
    top_domain = find_x_y_reorder(boundary=dict_tag_lines['Top'])
    bot_domain = find_x_y_reorder(boundary=[dict_tag_lines['Bottom_wed'], dict_tag_lines['Bottom_sla']])
    right_domain = find_x_y_reorder(boundary=[dict_tag_lines['Right_wed'], dict_tag_lines['Right_lit']],sorting_direction='y')
    left_domain = find_x_y_reorder(boundary=dict_tag_lines['Left_inlet'],sorting_direction='y')
    slab_top = find_x_y_reorder(boundary=[dict_tag_lines['Subduction_top_lit'],dict_tag_lines['Subduction_top_wed']])
    slab_bot =  find_x_y_reorder(boundary=dict_tag_lines['Subduction_bot'])
    ns_bnd = find_x_y_reorder(boundary=dict_tag_lines['Overriding_mantle'])

    b_mesh = boundary_mesh(top_surface_slab=slab_top
                           ,bot_surface_slab=slab_bot
                           ,ns_bond=ns_bnd
                           ,left_bd=left_domain
                           ,right_bd=right_domain
                           ,top_model=top_domain
                           ,bot_model=bot_domain)
    
    return b_mesh

def plt_boundary(ax,b_mesh:NamedTuple):
    for name, value in zip(b_mesh._fields, b_mesh):
        ax.plot(value[:,0],value[:,1],linewidth=0.8,c='k')
        
def modify_axis_pcolor_mesh(ax,label:list[str],label_bool:list[bool],x_tick_show:bool=True,y_tick_show:bool=True,letter:str='[a]')->None:
    ax.tick_params(axis="both", labelsize=_TICKSFONTSIZE_)      # tick labels
    # Hide unwanted spines
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Show and style top and left spines
    for spine in ["top", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("black")     # or any color
        ax.spines[spine].set_linewidth(1.5)

    # Put x-axis ticks on the top only
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    ax.tick_params(
        axis="both",
        direction="in",
        top=True,
        bottom=False,
        length=6,
        width=1.5,
        colors="black",
        pad = 3.5,
    )

    # Put y-axis ticks on the left only
    ax.yaxis.set_ticks_position("left")

    if label_bool[0]: 
        ax.set_xlabel(label[0],fontsize=_LABELFONTSIZE_ )
    if label_bool[1]:
        ax.set_ylabel(label[1],fontsize=_LABELFONTSIZE_ )
    if not x_tick_show:
        ax.set_xticklabels([])
    if not y_tick_show:
        ax.set_yticklabels([])
    ax.text(
        0.02, 0.97,letter,
        transform=ax.transAxes,
        c='w',
        fontweight='demibold',
        fontsize=10,
        va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='k', alpha=0.7)
    )
    
def extract_data_from_data_base(name_test:str,path)->tuple[float,float,int]:
    
    with h5py.File(f'{path}/benchmark_van_keken.h5') as f: 
        r_consv_a = f[f'{name_test}/r_cons_comb_r'][()]
        r_res_a = f[f'{name_test}/r_res_comb'][()]
            
    it_a = len(r_res_a[r_res_a!=0.0])

    return r_consv_a,r_res_a,it_a

current_dir = Path(__file__).resolve().parents[1]
test_name_a = current_dir/'VanKeken'/'T_vi2_th0'
test_name_b = current_dir/'VanKeken'/'T_vi2_th2'
test_a = Test(test_name_a)
test_b = Test(test_name_b)
temp_a = test_a.interpolate_data(Data_field='SteadyState.Temp')
temp_b = test_b.interpolate_data(Data_field='SteadyState.Temp')

b_mesh = extract_boundary_mesh(test=test_a)
r0a,r1a,ita = extract_data_from_data_base(name_test='T_vi2_th0',path=current_dir/'VanKeken')
r0b,r1b,itb = extract_data_from_data_base(name_test='T_vi2_th2',path=current_dir/'VanKeken')
# Extract the relevant boundaries: 
bx = 0.15 
by = 0.1 
sx = 0.35 
dx = 0.05 
sy = 0.35
dy = 0.12

# -- 
fig = plt.figure(figsize=[12,10])
# Prepare axis
ax0 = fig.add_axes([bx, by+sy+dy, sx, sy])
ax1 = fig.add_axes([bx+dx+sx, by+sy+dy, sx, sy])
ax2 = fig.add_axes([bx, by, sx, sy])
ax3 = fig.add_axes([bx+dx+sx, by, sx, sy])
ax4 = fig.add_axes([bx, 0.04, 0.7, 0.02])
levels = np.linspace(0,1300,14)
a=ax0.contourf(test_a.MeshData.Xi,test_a.MeshData.Yi,temp_a,cmap = 'cmc.lipari',alpha=0.6,levels=levels,  extend="max")
b=ax1.contourf(test_b.MeshData.Xi,test_b.MeshData.Yi,temp_b,cmap = 'cmc.lipari',alpha=0.6,levels=levels, extend="max")
plt_boundary(ax0,b_mesh=b_mesh)
plt_boundary(ax1,b_mesh=b_mesh)
modify_axis_pcolor_mesh(ax0,label=['x,[km]','y, [km]'],label_bool=[True,True])
modify_axis_pcolor_mesh(ax1,label=['x,[km]','y, [km]'],label_bool=[True,False],y_tick_show=False,letter='[b]')
ax2.plot(range(ita),r0a[0:ita],c='forestgreen',linewidth=1.6,label = 'real conservation laws')
ax2.plot(range(ita),r1a[0:ita],c='firebrick',linewidth=1.6, label = 'difference residual')
ax3.plot(range(itb),r0b[0:itb],c='forestgreen',linewidth=1.6,label = 'real conservation laws')
ax3.plot(range(itb),r1b[0:itb],c='firebrick',linewidth=1.6,label = 'difference residual')
ax3.axhline(1e-4,linewidth=1.0,c='b')
ax2.axhline(1e-4,linewidth=1.0,c='b')

ax2.set_yscale('log')
ax3.set_yscale('log')
modify_axis_pcolor_mesh(ax2,label=['it, [n.d.]',r'|res|, [n.d.]'],label_bool=[True,True],letter='[c]')
modify_axis_pcolor_mesh(ax3,label=['it, [n.d.]',r'|res|, [n.d.]'],label_bool=[True,False],y_tick_show=False,letter='[d]')
cb = plot_colorbar_and_deactivate(fig=fig,cax=ax4,mappable=a,orientation='horizontal',label=r'T,$[^{\circ}C]$')
fig.savefig(f'{current_dir}/benchmarks.png',transparent=True,dpi=600)
print('bla')




