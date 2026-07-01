"""_summary_
    General output folder-> where I write the functions required to print the file and databases
    
"""
    
from .utils import compute_strain_rate,compute_eii,print_ph
from .material_property.compute_material_property import compute_viscosity_FX,density_FX,heat_capacity_FX,heat_conductivity_FX,alpha_FX
import dolfinx.fem as fem
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import XDMFFile
from pathlib import Path
import os
import basix 
import h5py
from stonedfenicsx.config.scal import Scal
from stonedfenicsx.config.numerical_control import SimulationControls
from stonedfenicsx.config.geometry import Domain
from stonedfenicsx.config.phase_db import PhaseDataBase
from stonedfenicsx.solver_module.problems_solution import Solution
from stonedfenicsx.utils import print_ph, timing
from stonedfenicsx.material_property.compute_material_property import RHEOLOGYCACHED, THERMALCACHED




class OUTPUT(): 
    """
    This class handles output operations for the simulation. This class has been written partially asking chatgpt but then modified to fit my needs.
    """ 
    def __init__(self,
                 domain:Domain,
                 ctrl_sim:SimulationControls,
                 sc:Scal,
                 pdb:PhaseDataBase,
                 cach_mat_thermal:THERMALCACHED,
                 comm=MPI.COMM_WORLD):
        
        
        pt_save = ctrl_sim.ctrl_io.path_save
        if not os.path.exists(pt_save):
            os.makedirs(pt_save)
    
        pt_save = os.path.join(pt_save, ctrl_sim.ctrl_io.sname)
        if not os.path.exists(pt_save):
            os.makedirs(pt_save)    
        
        self.cached_mat_rheology = RHEOLOGYCACHED(pdb=pdb,phase=domain.phase)
        self.cached_mat_thermal = cach_mat_thermal # [Took the reference from problems, it is efficient to store the thermal properties in a class and then use it for the output]
        self.domain = domain
        self.pt_save = pt_save

        self.vel_V   = fem.functionspace(domain.mesh,
                          basix.ufl.element("Lagrange", "triangle", 1, shape=(domain.mesh.geometry.dim,)))
        self.pres_V  = fem.functionspace(domain.mesh,
                          basix.ufl.element("Lagrange", "triangle", 1))
        self.temp_V  = self.pres_V
        self.stress_V= fem.functionspace(domain.mesh,
                          basix.ufl.element("Lagrange", "triangle", 1, shape=(domain.mesh.geometry.dim**2,)))
        
        self.u_sol = fem.Function(self.vel_V)      
        self.u_sol.name = "Velocity  [cm/yr]"   
        self.p_dyn = fem.Function(self.pres_V)
        self.p_dyn.name = "Pressure  [GPa]"       
        self.temp_old = fem.Function(self.temp_V)
        self.temp_old.name = "Temperature old [degC]"
        self.temp_new = fem.Function(self.temp_V)
        self.temp_new.name = "Temperature  [degC]"
        self.p_lithostatic = fem.Function(self.temp_V)
        self.p_lithostatic.name = "Lit Pres  [GPa]"
        self.rho = fem.Function(self.temp_V)
        self.rho.name = "Density  [kg/m3]"
        self.cp = fem.Function(self.temp_V)
        self.cp.name = "Cp  [J/kg]"
        self.k = fem.Function(self.temp_V)
        self.k.name = "k  [W/m/K]"
        self.kappa = fem.Function(self.temp_V)
        self.kappa.name = "kappa  [m2/s]"
        self.alpha = fem.Function(self.temp_V)
        self.alpha.name = "alpha  [1/K]"
        self.e_ii = fem.Function(self.temp_V)
        self.e_ii.name = "Strain Rate  [1/s]"
        self.eta = fem.Function(self.temp_V)
        self.eta.name = "Viscosity  [Pa.s]"
        self.flux = fem.Function(self.vel_V) 
        self.flux.name = "Heat flux [W/m2]"      
        self.shear_heating = fem.Function(self.temp_V)
        self.shear_heating.name = "H_s [W/m3]" 
        self.tag = fem.Function(self.temp_V)
        self.tag.name = "MeshTAG"

        # for transient we keep XDMF files open across timesteps
        if ctrl_sim.ctrl.steady_state == 0:
            self.xdmf_main = XDMFFile(comm,
                                      os.path.join(self.pt_save, "time_dependent.xdmf"),
                                      "w")
    
            # write mesh once
            coord = domain.mesh.geometry.x.copy()
            domain.mesh.geometry.x[:] *= sc.length/1e3
            self.xdmf_main.write_mesh(domain.mesh)
            domain .mesh.geometry.x[:] = coord
            self.xdmf_main.close()

        else:
            self.xdmf_main = None
            
        
        
    # --- 
    def print_output(self
                     ,ctrl_sim:SimulationControls
                     ,sol:Solution              # Solution object
                     ,sc:Scal
                     ,it_outer:int = 0     # outer iteration counter
                     ,time:float=0.0,        # current time
                     ts:int=0,
                     debug:int=0)->None:            # time step counter        
        
        def interpolate_expression(fun, expr):
            fs = fun.function_space
            fun.interpolate(fem.Expression(expr,fs.element.interpolation_points()))

        
        self.u_sol.interpolate(sol.u_global)
        self.u_sol.x.array[:] = self.u_sol.x.array[:]*(sc.length/sc.time)/sc.scale_vel
        self.u_sol.x.scatter_forward()
        # Pressure
        self.p_dyn.interpolate(sol.p_global)
        self.p_dyn.x.array[:] = self.p_dyn.x.array[:]*sc.stress/1e9
        self.p_dyn.x.scatter_forward()
        # Temperature
        self.temp_old.interpolate(sol.T_O)
        self.temp_old.x.array[:] = self.temp_old.x.array[:]*sc.temp - 273.15
        self.temp_old.x.scatter_forward()
        
        self.temp_new.interpolate(sol.T_N)
        self.temp_new.x.array[:] = self.temp_new.x.array[:]*sc.temp - 273.15
        self.temp_new.x.scatter_forward()
        # Lithostatic pressure
        self.p_lithostatic.interpolate(sol.PL)
        self.p_lithostatic.x.array[:] = self.p_lithostatic.x.array[:]*sc.stress/1e9
        self.p_lithostatic.x.scatter_forward()
        
        # alpha 
        alpha = alpha_FX(self.cached_mat_thermal,sol.T_N,sol.PL)
        interpolate_expression(self.alpha, alpha)
        self.alpha.x.array[:] = self.alpha.x.array[:]*(1/sc.temp)
        self.alpha.x.scatter_forward()


        # density 
        rho = density_FX(self.cached_mat_thermal,sol.T_N,sol.PL)
        interpolate_expression(self.rho, rho)
        self.rho.x.array[:] = self.rho.x.array[:]*sc.rho
        self.rho.x.scatter_forward()

        # Cp 
        Cp = heat_capacity_FX(self.cached_mat_thermal,sol.T_N,sol.PL)
        interpolate_expression(self.cp, Cp)
        self.cp.x.array[:] = self.cp.x.array[:]*sc.cp
        self.cp.x.scatter_forward()

        # k 
        k = heat_conductivity_FX(self.cached_mat_thermal,sol.T_N,sol.PL,Cp,rho)
        interpolate_expression(self.k, k)
        self.k.x.array[:] = self.k.x.array[:]*sc.watt/(sc.length*sc.temp)
        self.k.x.scatter_forward()

        # kappa 
        self.kappa.x.array[:] = self.k.x.array[:]/(self.rho.x.array[:]*self.cp.x.array[:])
        self.kappa.x.scatter_forward()

        # strain rate 
        e = compute_strain_rate(sol.u_global)
        eII = compute_eII(e)
        interpolate_expression(self.e_ii, eII)
        self.e_ii.x.array[:] = np.abs(self.e_ii.x.array[:])*(1/sc.time)
        self.e_ii.x.array[self.e_ii.x.array[:] < 1e-20] = 0.0
        self.e_ii.x.scatter_forward()

        eta = compute_viscosity_FX(eII,sol.T_N,sol.PL,self.cached_mat_rheology,sc)
        interpolate_expression(self.eta, eta)
        self.eta.x.array[:] = self.eta.x.array[:]*sc.stress*sc.time 
        self.eta.x.scatter_forward()
        self.eta.x.scatter_forward()
        # heat flux 

        q_expr = - ( heat_conductivity_FX(self.cached_mat_thermal,sol.T_N,sol.PL,Cp,rho)* ufl.grad(sol.T_N))  
        interpolate_expression(self.flux, q_expr)   
        self.flux.x.array[:] = self.flux.x.array[:]*sc.watt/(sc.length**2)
        self.flux.x.scatter_forward()
        
        # shear heating
        self.shear_heating.interpolate(sol.Hs_global)
        self.shear_heating.x.array[:] = self.shear_heating.x.array[:]*sc.watt/(sc.length**3)
        self.shear_heating.x.scatter_forward()
        # Line Tag for the mesh and post_processing -> Translating in parallel -> gather and sending 
        tag = fem.Function(self.temp_V)
        tag.name = 'MeshTAG'
        
        boundary_list = np.array(list(self.domain.bc_dict.values()),dtype=np.int32)
        
        for i in range(len(self.domain.bc_dict.values())):
            facet_indices = self.domain.facets.find(boundary_list[i])   # numpy array of facet ids
            dofs = fem.locate_dofs_topological(self.temp_V, 1, facet_indices)
            tag.x.array[dofs] = boundary_list[i]
        
        tag.x.scatter_forward()
        
        self.tag.interpolate(tag)
        self.tag.x.scatter_forward()
        
        comm = self.domain.mesh.comm
        tdim = self.domain.mesh.topology.dim

        # DG0 on cells
        V0 = fem.functionspace(self.domain.mesh, ("DG", 0))
        part = fem.Function(V0, name="mpi_rank")

        # Which cells are owned by this rank?
        # cell index map: owned are [0, size_local)
        imap = self.domain.mesh.topology.index_map(tdim)
        n_local = imap.size_local

        # DG0 has one dof per cell, in the same ordering
        # (this is true for standard DG0 on cells in dolfinx)
        values = part.x.array
        values[:n_local] = comm.rank
        # ghosts (if present) can be left as-is; ParaView will still show partitioning
        part.x.scatter_forward()

        if ctrl_sim.ctrl.steady_state == 0:
            # transient: append to ongoing XDMF with time
            # write each field at this physical_time
            # ...same for PL2, rho2, Cp2, k2, kappa2, e_T, eta2, flux
            
            self.xdmf_main = XDMFFile(comm, os.path.join(self.pt_save, "time_dependent.xdmf"),
                                      "a")
            
            self.xdmf_main.write_function(self.u_sol,          time)
            self.xdmf_main.write_function(self.p_dyn,           time)
            self.xdmf_main.write_function(self.temp_new,           time)
            self.xdmf_main.write_function(self.temp_old,           time)
            self.xdmf_main.write_function(self.p_lithostatic,          time)
            self.xdmf_main.write_function(self.rho,         time)
            self.xdmf_main.write_function(self.cp,          time)
            self.xdmf_main.write_function(self.k,           time)
            self.xdmf_main.write_function(self.kappa,       time)
            self.xdmf_main.write_function(self.alpha,       time)
            self.xdmf_main.write_function(self.e_ii,          time)
            self.xdmf_main.write_function(self.eta,         time)
            self.xdmf_main.write_function(self.flux,         time)
            self.xdmf_main.write_function(self.shear_heating,time)
            self.xdmf_main.write_function(self.tag,          time)
            
            self.close()
        
        else:
            with XDMFFile(MPI.COMM_WORLD, os.path.join(self.pt_save, "Steady_State.xdmf"), "w") as ufile_xdmf:
                print_ph('... Printing')
                coord = self.domain.mesh.geometry.x.copy()
                self.domain.mesh.geometry.x[:] *= sc.length/1e3
                ufile_xdmf.write_mesh(self.domain.mesh)
                ufile_xdmf.write_function(self.u_sol)
                ufile_xdmf.write_function(self.p_dyn)
                ufile_xdmf.write_function(self.temp_new)
                ufile_xdmf.write_function(self.temp_old)
                ufile_xdmf.write_function(self.p_lithostatic)
                ufile_xdmf.write_function(self.rho)
                ufile_xdmf.write_function(self.cp)
                ufile_xdmf.write_function(self.k)
                ufile_xdmf.write_function(self.kappa)
                ufile_xdmf.write_function(self.alpha)
                ufile_xdmf.write_function(self.e_ii)
                ufile_xdmf.write_function(self.eta)
                ufile_xdmf.write_function(self.flux)
                ufile_xdmf.write_function(self.shear_heating)
                ufile_xdmf.write_function(self.tag)
                ufile_xdmf.write_function(part)
                self.domain.mesh.geometry.x[:] = coord
                print_ph('... Finished')
    
    def close(self):
    
        if self.xdmf_main is not None:
            self.xdmf_main.close()
            


#-----------------------------------------------------------------------------------------
# Benchmarking functions    
#-----------------------------------------------------------------------------------------
def _benchmark_van_keken(sol:Solution
                         ,ctrl_sim:SimulationControls
                         ,sc:Scal)->None:
    from scipy.interpolate import griddata
    
    comm = MPI.COMM_WORLD


    lT = sol.T_N.copy()
    fs = lT.function_space
    imap = fs.dofmap.index_map
    n_owned = imap.size_local * fs.dofmap.bs  
    array = lT.x.array[:n_owned].copy()

    # gather solution from all processes on proc 0
    gT = comm.gather(array, root=0)
    
    XGl = sol.T_N.function_space.tabulate_dof_coordinates()#gather_coordinates(sol.T_O.function_space)
    x  = XGl[:n_owned,0]
    y  = XGl[:n_owned,1]
    X_G = comm.gather(x,root=0)
    Y_G = comm.gather(y,root=0)
    
    if comm.rank == 0:
        T_all = np.concatenate(gT)
        X_global = np.concatenate(X_G)
        Y_global = np.concatenate(Y_G)
        
        
        
        XG = np.zeros([len(X_global),2])
        XG[:,0] = X_global
        XG[:,1] = Y_global
     
        x_g = np.array([np.min(XG[:,0]),np.max(XG[:,0])],dtype=np.float64)*sc.L
        y_g = np.array([np.min(XG[:,1]),np.max(XG[:,1])],dtype=np.float64)*sc.L
        nx   = 111 
        ny   = 101
        idx0 = 10 #(11 in van keken)
        idy0 = ny-11 


        xx   = np.linspace(x_g[0],x_g[1],num=nx)
        yy   = np.linspace(y_g[0],y_g[1],num=ny)
        T_g  = np.zeros([nx,ny],dtype=np.float64)
        X,Y  = np.meshgrid(xx,yy)
        xt,yt = XG[:,0]*sc.L, XG[:,1]*sc.L
        p     = np.zeros((len(xt),2),dtype=np.float64)
        p[:,0] = xt 
        p[:,1] = yt
        T_g   = griddata(p,T_all*sc.Temp,(X, Y), method='nearest')-273.15
        T_g = T_g.T
        T = 0
        co = 0
        x_s=[]
        y_s=[] 
        T2 = []
        c = 0
        X_S = []
        Y_S = []
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

        print( '------------------------------------------------------------------' )
        print( ':::====> T(11,11) = %.2f [deg C]'%( T_11_11, ) )
        print( ':::====> L2_A     = %.2f [deg C]'%( T_ln,     ) )
        print( ':::====> L2_B     = %.2f [deg C]'%( T_ln2,   ) )
        print( ':::L2_A = T along the slab surface from 0 to -210 km' )
        print( ':::L2_B = T wedge norm from -54 to -110 km ' )
        print( ':::=> From grid to downsampled grid -> nearest interpolation' )
        print( '------------------------------------------------------------------' )
        # place holder for the save database 
        
    if comm.rank ==0:
        
        with h5py.File(os.path.join(ctrl_sim.ctrl_io.path_save,'benchmark_van_keken.h5'),'a') as van_keken_db:
        
           group_name = ctrl_sim.ctrl_io.sname
           name       = '%s/T_11_11'%group_name
           if name in van_keken_db.keys():
               del van_keken_db[name]
           
           van_keken_db.create_dataset(name,data=T_11_11)
           
           name       = '%s/L2_A'%group_name
           
           if name in van_keken_db.keys():
               del van_keken_db[name]
           van_keken_db.create_dataset(name,data=T_ln)
           
           name       = '%s/L2_B'%group_name
           
           if name in van_keken_db.keys():
               del van_keken_db[name]
           van_keken_db.create_dataset(name,data=T_ln2)
           
           name = '%s/Outer_residual'%group_name
           if name in van_keken_db.keys():
               del van_keken_db[name]
           van_keken_db.create_dataset(name,data = sol.outer_iteration)
           
           name = '%s/maxTemperature'%group_name 
           if name in van_keken_db.keys():
               del van_keken_db[name]
           van_keken_db.create_dataset(name,data = sol.MT)
           
           name = '%s/minTemperature'%group_name 
           if name in van_keken_db.keys():
               del van_keken_db[name]
           van_keken_db.create_dataset(name,data = sol.mT)
           
           name = '%s/RMST'%group_name 
           if name in van_keken_db.keys():
               del van_keken_db[name]
           van_keken_db.create_dataset(name,data = sol.RMST)
           
           name = '%s/maxVel'%group_name 
           if name in van_keken_db.keys():
               del van_keken_db[name]
           van_keken_db.create_dataset(name,data = sol.Mv)
           
           name = '%s/minVel'%group_name 
           if name in van_keken_db.keys():
               del van_keken_db[name]
           van_keken_db.create_dataset(name,data = sol.mv)
           
           name = '%s/RMSv'%group_name 
           if name in van_keken_db.keys():
               del van_keken_db[name]
           van_keken_db.create_dataset(name,data = sol.RMSv)
           
           name = '%s/ts'%group_name 
           if name in van_keken_db.keys():
               del van_keken_db[name]
           van_keken_db.create_dataset(name,data = sol.ts)        


