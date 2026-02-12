"""_summary_
    General output folder-> where I write the functions required to print the file and databases
    
"""
    
from .package_import import *
from .utils import evaluate_material_property,compute_strain_rate,compute_eII,print_ph
from .material_property.compute_material_property import compute_viscosity_FX,density_FX,heat_capacity_FX,heat_conductivity_FX,alpha_FX



class OUTPUT(): 
    """
    This class handles output operations for the simulation. This class has been written partially asking chatgpt but then modified to fit my needs.
    """ 
    def __init__(self, mesh, ioctrl, ctrl, sc, comm=MPI.COMM_WORLD):
        
        
        pt_save = ioctrl.path_save
        if not os.path.exists(pt_save):
            os.makedirs(pt_save)
    
        pt_save = os.path.join(pt_save, ioctrl.sname)
        if not os.path.exists(pt_save):
            os.makedirs(pt_save)    
        
        self.pt_save = pt_save

        
        import basix
        
        self.vel_V   = fem.functionspace(mesh.mesh,
                          basix.ufl.element("Lagrange", "triangle", 1, shape=(mesh.mesh.geometry.dim,)))
        self.pres_V  = fem.functionspace(mesh.mesh,
                          basix.ufl.element("Lagrange", "triangle", 1))
        self.temp_V  = self.pres_V
        self.stress_V= fem.functionspace(mesh.mesh,
                          basix.ufl.element("Lagrange", "triangle", 1, shape=(mesh.mesh.geometry.dim**2,)))

        # for transient we keep XDMF files open across timesteps
        if ctrl.steady_state == 0:
            self.xdmf_main = XDMFFile(comm,
                                      os.path.join(self.pt_save, "time_dependent.xdmf"),
                                      "w")
            # write mesh once
            coord = mesh.mesh.geometry.x.copy()
            mesh.mesh.geometry.x[:] *= sc.L/1e3
            self.xdmf_main.write_mesh(mesh.mesh)
            mesh.mesh.geometry.x[:] = coord

        else:
            self.xdmf_main = None
        
        
    
    
    def print_output(self
                     ,S                # Solution object
                     ,D                # Global domain object
                     ,FGT              # Phase database object
                     ,FGR
                     ,ioctrl           # IO control object
                     ,sc               # Scaling object
                     ,ctrl             # Numerical control object
                     ,it_outer = 0     # outer iteration counter
                     ,time=0.0,        # current time
                     ts=0):            # time step counter
        import os
        from dolfinx.io import XDMFFile
        if ctrl.steady_state == 1:
            file_name = os.path.join(self.pt_save,'Steady_state')         
        file_name2 = os.path.join(self.pt_save,'MeshTag')
        print_ph('... Velocity')
        # Velocity 
        u_T = fem.Function(self.vel_V)
        u_T.name = "Velocity  [cm/yr]"
        u_T.interpolate(S.u_global)
        u_T.x.array[:] = u_T.x.array[:]*(sc.L/sc.T)/sc.scale_vel
        u_T.x.scatter_forward()
        print_ph('... Pressure')
        # Pressure
        p2 = fem.Function(self.temp_V)
        p2.name = "Pressure  [GPa]"
        p2.interpolate(S.p_global)
        p2.x.array[:] = p2.x.array[:]*sc.stress/1e9 
        p2.x.scatter_forward()  
        print_ph('... Temperature')
        # Temperature
        T2 = fem.Function(self.temp_V)
        T2.name = "Temperature  [degC]"
        if ctrl.steady_state == 1:
            T2.interpolate(S.T_N)
        else:
            T2.interpolate(S.T_N)
        T2.x.array[:] = T2.x.array[:]*sc.Temp - 273.15
        T2.x.scatter_forward()  

        # Temperature
        T3 = fem.Function(self.temp_V)
        T3.name = "Temperature old [degC]"
        T3.interpolate(S.T_O)
        T3.x.array[:] = T3.x.array[:]*sc.Temp - 273.15
        T3.x.scatter_forward()  
        print_ph('... Lithostatic')
        # Lithostatic pressure
        PL2 = fem.Function(self.temp_V)
        PL2.name = "Lit Pres  [GPa]"
        PL2.interpolate(S.PL)
        PL2.x.array[:] = PL2.x.array[:]*sc.stress/1e9
        PL2.x.scatter_forward()
        
        # alpha 
        print_ph('... Thermal Expansivity')
        alpha = alpha_FX(FGT,S.T_N,S.PL)
        alpha2 = evaluate_material_property(alpha, self.temp_V)
        alpha2.name = "alpha  [1/K]"
        alpha2.x.array[:] = alpha2.x.array[:]*(1/sc.Temp)
        alpha2.x.scatter_forward()


        # density 
        print_ph('...Density ')
        rho = density_FX(FGT,S.T_N,S.PL)
        rho2 = evaluate_material_property(rho, self.temp_V)
        rho2.name = "Density  [kg/m3]"
        rho2.x.array[:] = rho2.x.array[:]*sc.rho
        rho2.x.scatter_forward()

        # Cp 
        print_ph('... Capacity')
        Cp = heat_capacity_FX(FGT,S.T_N)
        Cp2 = evaluate_material_property(Cp,self.temp_V)
        Cp2.name = "Cp  [J/kg]"
        Cp2.x.array[:] = Cp2.x.array[:]*sc.Cp
        Cp2.x.scatter_forward()

        # k 
        print_ph('... Conductivity')
        k = heat_conductivity_FX(FGT,S.T_N,S.PL,Cp,rho)
        k2 = evaluate_material_property(k,self.temp_V)
        k2.name = "k  [W/m/k]"
        k2.x.array[:] = k2.x.array[:]*sc.k
        k2.x.scatter_forward()


        # kappa 
        print_ph('... Diffusivity')
        kappa2 = fem.Function(self.temp_V)
        kappa2.name = "kappa  [m2/s]"
        kappa2.x.array[:] = k2.x.array[:]/(rho2.x.array[:]*Cp2.x.array[:])
        kappa2.x.scatter_forward()

        # strain rate 
        print_ph('... Strain Rate')
        e = compute_strain_rate(S.u_global)
        eII = compute_eII(e)
        eII2 = evaluate_material_property(eII, self.temp_V)

        e_T = fem.Function(self.temp_V)
        e_T.name = "e_II  [1/s]"
        e_T.interpolate(eII2)
        e_T.x.array[:] = e_T.x.array[:]/sc.T
        e_T.x.array[e_T.x.array[:]<=1e-22] = 1e-20
        e_T.x.scatter_forward()

        # viscosity (e,S.t_oslab,S.p_lslab,pdb,D.phase,D,sc)
        print_ph('... Viscosity')

        eta = compute_viscosity_FX(eII,S.T_N,S.PL,FGR,sc)
        eta2 = evaluate_material_property(eta,self.temp_V)
        eta2.name = "eta  [Pa s]"
        eta2.x.array[:] = np.abs(eta2.x.array[:])*sc.eta
        eta2.x.scatter_forward()

        # heat flux 
        print_ph('... Flux')

        q_expr = - ( heat_conductivity_FX(FGT,S.T_O,S.PL,Cp,rho)* ufl.grad(S.T_N))  
        flux = evaluate_material_property(q_expr,self.vel_V)
        flux.name = 'Heat flux [W/m2]'
        flux.x.array[:] *= sc.Watt/sc.L**2
        
        print_ph('... MeshTag')
        # Line Tag for the mesh and post_processing -> Translating in parallel -> gather and sending 
        tag = fem.Function(self.temp_V)
        tag.name = 'MeshTAG'
        
        boundary_list = np.array(list(D.bc_dict.values()),dtype=np.int32)
        
        for i in range(len(D.bc_dict.values())):
            facet_indices = D.facets.find(boundary_list[i])   # numpy array of facet ids
            dofs = fem.locate_dofs_topological(self.temp_V, 1, facet_indices)
            tag.x.array[dofs] = boundary_list[i]
        
        tag.x.scatter_forward()
        
        print_ph('... MeshTag')

        if ctrl.steady_state == 0:
            # transient: append to ongoing XDMF with time
            # write each field at this physical_time
            # ...same for PL2, rho2, Cp2, k2, kappa2, e_T, eta2, flux
            self.xdmf_main.write_function(u_T,          time)
            self.xdmf_main.write_function(p2,           time)
            self.xdmf_main.write_function(T2,           time)
            self.xdmf_main.write_function(T3,           time)
            self.xdmf_main.write_function(PL2,          time)
            self.xdmf_main.write_function(rho2,         time)
            self.xdmf_main.write_function(Cp2,          time)
            self.xdmf_main.write_function(k2,           time)
            self.xdmf_main.write_function(kappa2,       time)
            self.xdmf_main.write_function(alpha2,       time)
            self.xdmf_main.write_function(e_T,          time)
            self.xdmf_main.write_function(eta2,         time)
            self.xdmf_main.write_function(flux,         time)
            self.xdmf_main.write_function(tag,          time)
        else:
            with XDMFFile(MPI.COMM_WORLD, "%s.xdmf"%file_name, "w") as ufile_xdmf:
                print_ph('... Printing')
                coord = D.mesh.geometry.x.copy()
                D.mesh.geometry.x[:] *= sc.L/1e3
                ufile_xdmf.write_mesh(D.mesh)
                ufile_xdmf.write_function(u_T  )
                ufile_xdmf.write_function(p2   )
                ufile_xdmf.write_function(T2   )
                ufile_xdmf.write_function(T3   )
                ufile_xdmf.write_function(PL2  )
                ufile_xdmf.write_function(rho2 )
                ufile_xdmf.write_function(Cp2  )
                ufile_xdmf.write_function(k2   )
                ufile_xdmf.write_function(kappa2)
                ufile_xdmf.write_function(alpha2)
                ufile_xdmf.write_function(e_T  )
                ufile_xdmf.write_function(eta2 )
                ufile_xdmf.write_function(flux )
                ufile_xdmf.write_function(tag )
                D.mesh.geometry.x[:] = coord
                print_ph('... Finished')




        return 0
    
    def close(self):
    
        if self.xdmf_main is not None:
            self.xdmf_main.close()
            


#-----------------------------------------------------------------------------------------
# Benchmarking functions    
#-----------------------------------------------------------------------------------------
def _benchmark_van_keken(S,ctrl_io,sc):
    import h5py 
    from scipy.interpolate import griddata
    
    comm = MPI.COMM_WORLD

    lT = S.T_N.copy()
    mpi_comm = lT.function_space.mesh.comm
    array = lT.x.array

    # gather solution from all processes on proc 0
    gT = mpi_comm.gather(array, root=0)
    
    XGl = S.T_N.function_space.tabulate_dof_coordinates()#gather_coordinates(S.T_O.function_space)
    x  = XGl[:,0]
    y  = XGl[:,1]
    X_G = mpi_comm.gather(x,root=0)
    Y_G = mpi_comm.gather(y,root=0)
    
    if comm.rank == 0:
        XG = np.zeros([len(X_G[0]),2])
        XG[:,0] = X_G[0]
        XG[:,1] = Y_G[0] 

        T = gT[0]       
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
        T_g   = griddata(p,T*sc.Temp,(X, Y), method='nearest')-273.15
        T_g = T_g.transpose()
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
        if os.path.join(ctrl_io.path_save,'benchmark_van_keken.h5') :
            van_keken_db = h5py.File(os.path.join(ctrl_io.path_save,'benchmark_van_keken.h5'),'a')
        else: 
            van_keken_db = h5py.File(os.path.join(ctrl_io.path_save,'benchmark_van_keken.h5'),'w')
        
        group_name = ctrl_io.sname
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
        van_keken_db.create_dataset(name,data = S.outer_iteration)
        
        name = '%s/maxTemperature'%group_name 
        if name in van_keken_db.keys():
            del van_keken_db[name]
        van_keken_db.create_dataset(name,data = S.MT)
        
        name = '%s/minTemperature'%group_name 
        if name in van_keken_db.keys():
            del van_keken_db[name]
        van_keken_db.create_dataset(name,data = S.mT)
        
        name = '%s/ts'%group_name 
        if name in van_keken_db.keys():
            del van_keken_db[name]
        van_keken_db.create_dataset(name,data = S.ts)        

        van_keken_db.close()

    return 0 
