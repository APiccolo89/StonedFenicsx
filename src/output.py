"""_summary_
    General output folder-> where I write the functions required to print the file and databases
    
"""
    
from .package_import import *
from .utils import evaluate_material_property,compute_strain_rate,compute_eII,print_ph
from .compute_material_property import compute_viscosity_FX,density_FX,heat_capacity_FX,heat_conductivity_FX,alpha_FX



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
            coord = mesh.geometry.x.copy()
            mesh.geometry.x[:] *= sc.L/1e3
            self.xdmf_main.write_mesh(mesh)
            mesh.geometry.x[:] = coord

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
        import basix  

        if ctrl.steady_state == 1:
            file_name = os.path.join(self.pt_save,'Steady_state')
            
        file_name2 = os.path.join(self.pt_save,'MeshTag')



        # Prepare variable and spaces 


        # Velocity 
        u_T = fem.Function(self.vel_V)
        u_T.name = "Velocity  [cm/yr]"
        u_T.interpolate(S.u_global)
        u_T.x.array[:] = u_T.x.array[:]*(sc.L/sc.T)/sc.scale_vel
        u_T.x.scatter_forward()

        # Pressure
        p2 = fem.Function(self.temp_V)
        p2.name = "Pressure  [GPa]"
        p2.interpolate(S.p_global)
        p2.x.array[:] = p2.x.array[:]*sc.stress/1e9 
        p2.x.scatter_forward()  

        # Temperature
        T2 = fem.Function(self.temp_V)
        T2.name = "Temperature  [degC]"
        if ctrl.steady_state == 1:
            T2.interpolate(S.T_O)
        else:
            T2.interpolate(S.T_N)
        T2.x.array[:] = T2.x.array[:]*sc.Temp - 273.15
        T2.x.scatter_forward()  

        #if ctrl.steady_state == 0:
        dT = fem.Function(self.temp_V)
        dT.name = "dT  [degC]"
        a = S.T_N.copy()
        a.x.array[:] = a.x.array[:]-S.T_O.x.array[:]
        a.x.scatter_forward() 
        dT.interpolate(a)
        dT.x.array[:] = dT.x.array[:]*sc.Temp
        dT.x.scatter_forward()
        # Lithostatic pressure
        PL2 = fem.Function(self.temp_V)
        PL2.name = "Lit Pres  [GPa]"
        PL2.interpolate(S.PL)
        PL2.x.array[:] = PL2.x.array[:]*sc.stress/1e9
        PL2.x.scatter_forward()
        
        # alpha 
        alpha = alpha_FX(FGT,S.T_O,S.PL)
        alpha2 = evaluate_material_property(alpha, self.temp_V)
        alpha2.name = "alpha  [1/K]"
        alpha2.x.array[:] = alpha2.x.array[:]*(1/sc.Temp)
        alpha2.x.scatter_forward()


        # density 
        rho = density_FX(FGT,S.T_O,S.PL)
        rho2 = evaluate_material_property(rho, self.temp_V)
        rho2.name = "Density  [kg/m3]"
        rho2.x.array[:] = rho2.x.array[:]*sc.rho
        rho2.x.scatter_forward()

        # Cp 
        Cp = heat_capacity_FX(FGT,S.T_O)
        Cp2 = evaluate_material_property(Cp,self.temp_V)
        Cp2.name = "Cp  [J/kg]"
        Cp2.x.array[:] = Cp2.x.array[:]*sc.Cp
        Cp2.x.scatter_forward()

        # k 
        k = heat_conductivity_FX(FGT,S.T_O,S.PL,Cp,rho)
        k2 = evaluate_material_property(k,self.temp_V)
        k2.name = "k  [W/m/k]"
        k2.x.array[:] = k2.x.array[:]*sc.k
        k2.x.scatter_forward()


        # kappa 
        kappa2 = fem.Function(self.temp_V)
        kappa2.name = "kappa  [m2/s]"
        kappa2.x.array[:] = k2.x.array[:]/(rho2.x.array[:]*Cp2.x.array[:])
        kappa2.x.scatter_forward()

        # strain rate 
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
        eta = compute_viscosity_FX(eII,S.T_O,S.PL,FGR,sc)
        eta2 = evaluate_material_property(eta,self.temp_V)
        eta2.name = "eta  [Pa s]"
        eta2.x.array[:] = np.abs(eta2.x.array[:])*sc.eta
        eta2.x.scatter_forward()



        # heat flux 
        q_expr = - ( heat_conductivity_FX(FGT,S.T_O,S.PL,Cp,rho)* ufl.grad(S.T_O))  
        flux = evaluate_material_property(q_expr,self.vel_V)
        flux.name = 'Heat flux [W/m2]'
        flux.x.array[:] *= sc.Watt/sc.L**2
        # adiabatic heating 
        if ctrl.adiabatic_heating >0:
            adiabatic_expr = alpha * ufl.inner(S.u_wedge, ufl.grad(S.p_lwedge)) * S.t_owedge
            adiabatic_H = evaluate_material_property(adiabatic_expr,self.temp_V)
            adiabaticH = fem.Function(self.temp_V)
            adiabaticH.name = 'Ha [W/m3]'
            adiabaticH.x.array[:] = adiabatic_H.x.array[:]*sc.Watt/sc.L**3
        
            shear_heatingF =  fem.Function(self.temp_V)
            shear_heatingF.interpolate(S.Hs_global)
            shear_heatingF.name = 'Hs [W/m3]'
            shear_heatingF.x.array[:]*=sc.Watt/sc.L**3

            adiabatic_comp    = fem.Function(self.temp_V)
            adiabatic_comp.name = " Ha+Hs [W/m3]"
            adiabatic_comp.x.array[:] = adiabaticH.x.array[:] + shear_heatingF.x.array[:]
            adiabatic_comp.x.scatter_forward()
        else: 
            adiabaticH = fem.Function(self.temp_V)
            adiabaticH.name = 'Ha [W/m3]'
            adiabaticH.x.array[:] = 0.0 
            
            shear_heatingF =  fem.Function(self.temp_V)
            shear_heatingF.name = 'Hs [W/m3]'
            shear_heatingF.x.array[:] = 0.0 

            adiabatic_comp    = fem.Function(self.temp_V)
            adiabatic_comp.name = "Ha+Hs [W/m3]"
            adiabatic_comp.x.array[:] = 0.0 
            adiabatic_comp.x.scatter_forward()    

        
        # Line Tag for the mesh and post_processing 
        tag = fem.Function(self.temp_V)
        tag.name = 'MeshTAG'
        marker_unique = np.unique(D.facets.values)
        for i in range(len(marker_unique)):
            facet_indices = D.facets.find(marker_unique[i])   # numpy array of facet ids
            dofs = fem.locate_dofs_topological(self.temp_V, 1, facet_indices)
            tag.x.array[dofs] = marker_unique[i]
            tag.x.scatter_forward()
        
        with XDMFFile(MPI.COMM_WORLD, "%s.xdmf"%file_name2, "w") as ufile_xdmf:
        
            coord = D.mesh.geometry.x.copy()
            D.mesh.geometry.x[:] *= sc.L/1e3
            ufile_xdmf.write_mesh(D.mesh)
            ufile_xdmf.write_function(tag)
            D.mesh.geometry.x[:] = coord


        if ctrl.steady_state == 0:
            # transient: append to ongoing XDMF with time
            # write each field at this physical_time
            # ...same for PL2, rho2, Cp2, k2, kappa2, e_T, eta2, flux
            self.xdmf_main.write_function(dT,           time)
            self.xdmf_main.write_function(u_T,          time)
            self.xdmf_main.write_function(p2,           time)
            self.xdmf_main.write_function(T2,           time)
            self.xdmf_main.write_function(PL2,          time)
            self.xdmf_main.write_function(rho2,         time)
            self.xdmf_main.write_function(Cp2,          time)
            self.xdmf_main.write_function(k2,           time)
            self.xdmf_main.write_function(kappa2,       time)
            self.xdmf_main.write_function(alpha2,       time)
            self.xdmf_main.write_function(e_T,          time)
            self.xdmf_main.write_function(eta2,         time)
            self.xdmf_main.write_function(flux,         time)
            self.xdmf_main.write_function(adiabaticH,   time)
            self.xdmf_main.write_function(shear_heatingF,   time)
            self.xdmf_main.write_function(tag,          time)
        else:
            with XDMFFile(MPI.COMM_WORLD, "%s.xdmf"%file_name, "w") as ufile_xdmf:

                coord = D.mesh.geometry.x.copy()
                D.mesh.geometry.x[:] *= sc.L/1e3
                ufile_xdmf.write_mesh(D.mesh)
                ufile_xdmf.write_function(dT   )
                ufile_xdmf.write_function(u_T  )
                ufile_xdmf.write_function(p2   )
                ufile_xdmf.write_function(T2   )
                ufile_xdmf.write_function(PL2  )
                ufile_xdmf.write_function(rho2 )
                ufile_xdmf.write_function(Cp2  )
                ufile_xdmf.write_function(k2   )
                ufile_xdmf.write_function(kappa2)
                ufile_xdmf.write_function(alpha2)
                ufile_xdmf.write_function(e_T  )
                ufile_xdmf.write_function(eta2 )
                ufile_xdmf.write_function(flux )
                ufile_xdmf.write_function(adiabaticH )
                ufile_xdmf.write_function(shear_heatingF)
                ufile_xdmf.write_function(tag )
                D.mesh.geometry.x[:] = coord



        return 0
    
    def close(self):
    
        if self.xdmf_main is not None:
            self.xdmf_main.close()
            

class OUTPUT_WEDGE(): 
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
        
        self.vel_V   = fem.functionspace(mesh,
                          basix.ufl.element("Lagrange", "triangle", 1, shape=(mesh.geometry.dim,)))
        self.pres_V  = fem.functionspace(mesh,
                          basix.ufl.element("Lagrange", "triangle", 1))
        self.temp_V  = self.pres_V
        self.stress_V= fem.functionspace(mesh,
                          basix.ufl.element("Lagrange", "triangle", 1, shape=(mesh.geometry.dim**2,)))

        # for transient we keep XDMF files open across timesteps
        if ctrl.steady_state == 0:
            self.xdmf_main = XDMFFile(comm,
                                      os.path.join(self.pt_save, "time_dependent.xdmf"),
                                      "w")
            # write mesh once
            coord = mesh.geometry.x.copy()
            mesh.geometry.x[:] *= sc.L/1e3
            self.xdmf_main.write_mesh(mesh)
            mesh.geometry.x[:] = coord

        else:
            self.xdmf_main = None
        
        
    
    
    def print_output(self
                     ,S                # Solution object
                     ,D                # Global domain object
                     ,pdb              # Phase database object
                     ,ioctrl           # IO control object
                     ,sc               # Scaling object
                     ,ctrl             # Numerical control object
                     ,it_outer = 0     # outer iteration counter
                     ,time=0.0,        # current time
                     ts=0):            # time step counter
        
        import os
        from dolfinx.io import XDMFFile
        import basix  
        from .compute_material_property import density_FX,heat_capacity_FX,heat_conductivity_FX,alpha_FX
        if ctrl.steady_state == 1:
            file_name = os.path.join(self.pt_save,'Steady_stateWG')
            
        file_name2 = os.path.join(self.pt_save,'MeshTag')



        # Prepare variable and spaces 


        # Velocity 
        u_T = fem.Function(self.vel_V)
        u_T.name = "Velocity  [cm/yr]"
        u_T.interpolate(S.u_wedge)
        u_T.x.array[:] = u_T.x.array[:]*(sc.L/sc.T)/sc.scale_vel
        u_T.x.scatter_forward()

        # Pressure
        p2 = fem.Function(self.temp_V)
        p2.name = "Pressure  [GPa]"
        p2.interpolate(S.p_wedge)
        p2.x.array[:] = p2.x.array[:]*sc.stress/1e9 
        p2.x.scatter_forward()  

        # Temperature
        T2 = fem.Function(self.temp_V)
        T2.name = "Temperature  [degC]"
        if ctrl.steady_state == 1:
            T2.interpolate(S.t_owedge)
        else:
            T2.interpolate(S.T_N)
        T2.x.array[:] = T2.x.array[:]*sc.Temp - 273.15
        T2.x.scatter_forward()  

   
        # Lithostatic pressure
        PL2 = fem.Function(self.temp_V)
        PL2.name = "Lit Pres  [GPa]"
        PL2.interpolate(S.p_lwedge)
        PL2.x.array[:] = PL2.x.array[:]*sc.stress/1e9
        PL2.x.scatter_forward()
        
        # alpha 
        alpha = alpha_FX(pdb,S.t_owedge,S.p_lwedge,D.phase,D)
        alpha2 = evaluate_material_property(alpha, self.temp_V)
        alpha2.name = "alpha  [1/K]"
        alpha2.x.array[:] = alpha2.x.array[:]*(1/sc.Temp)
        alpha2.x.scatter_forward()


        # density 
        rho = density_FX(pdb,S.t_owedge,S.p_lwedge,D.phase,D)
        rho2 = evaluate_material_property(rho, self.temp_V)
        rho2.name = "Density  [kg/m3]"
        rho2.x.array[:] = rho2.x.array[:]*sc.rho
        rho2.x.scatter_forward()

        # Cp 
        Cp = heat_capacity_FX(pdb,S.t_owedge,D.phase,D)
        Cp2 = evaluate_material_property(Cp,self.temp_V)
        Cp2.name = "Cp  [J/kg]"
        Cp2.x.array[:] = Cp2.x.array[:]*sc.Cp
        Cp2.x.scatter_forward()

        # k 
        k = heat_conductivity_FX(pdb,S.t_owedge,S.p_lwedge,D.phase,D,Cp,rho)
        k2 = evaluate_material_property(k,self.temp_V)
        k2.name = "k  [W/m/k]"
        k2.x.array[:] = k2.x.array[:]*sc.k
        k2.x.scatter_forward()


        # kappa 
        kappa2 = fem.Function(self.temp_V)
        kappa2.name = "kappa  [m2/s]"
        kappa2.x.array[:] = k2.x.array[:]/(rho2.x.array[:]*Cp2.x.array[:])
        kappa2.x.scatter_forward()

        # strain rate 
        e             = compute_strain_rate(S.u_wedge)
        
        
        eII = compute_eII(e)

        
        
        eII2 = evaluate_material_property(eII, self.temp_V)

        e_T = fem.Function(self.temp_V)
        e_T.name = "e_II  [1/s]"
        e_T.interpolate(eII2)
        e_T.x.array[:] = e_T.x.array[:]/sc.T
        e_T.x.array[e_T.x.array[:]<=1e-22] = 1e-20
        e_T.x.scatter_forward()

        # viscosity (e,S.t_oslab,S.p_lslab,pdb,D.phase,D,sc)
        eta = compute_viscosity_FX(eII,S.t_owedge,S.p_lwedge,pdb,D.phase,D,sc)
        eta2 = evaluate_material_property(eta,self.temp_V)
        eta2.name = "eta  [Pa s]"
        eta2.x.array[:] = np.abs(eta2.x.array[:])*sc.eta
        eta2.x.scatter_forward()

        # adiabatic heating 
        if ctrl.adiabatic_heating >0:
            adiabatic_expr = alpha * ufl.inner(S.u_wedge, ufl.grad(S.p_lwedge)) * S.t_owedge
            adiabatic_H = evaluate_material_property(adiabatic_expr,self.temp_V)
            adiabaticH = fem.Function(self.temp_V)
            adiabaticH.name = 'Ha [W/m3]'
            adiabaticH.x.array[:] = adiabatic_H.x.array[:]*sc.Watt/sc.L**3
        
            shear_heatingF =  fem.Function(self.temp_V)
            shear_heatingF.interpolate(S.Hs_global)
            shear_heatingF.name = 'Hs [W/m3]'
            shear_heatingF.x.array[:]*=sc.Watt/sc.L**3

            adiabatic_comp    = fem.Function(self.temp_V)
            adiabatic_comp.name = " Ha+Hs [W/m3]"
            adiabatic_comp.x.array[:] = adiabaticH.x.array[:] + shear_heatingF.x.array[:]
            adiabatic_comp.x.scatter_forward()
        else: 
            adiabaticH = fem.Function(self.temp_V)
            adiabaticH.name = 'Ha [W/m3]'
            adiabaticH.x.array[:] = 0.0 
            
            shear_heatingF =  fem.Function(self.temp_V)
            shear_heatingF.name = 'Hs [W/m3]'
            shear_heatingF.x.array[:] = 0.0 

            adiabatic_comp    = fem.Function(self.temp_V)
            adiabatic_comp.name = "Ha+Hs [W/m3]"
            adiabatic_comp.x.array[:] = 0.0 
            adiabatic_comp.x.scatter_forward()    
        
        # Line Tag for the mesh and post_processing 
        tag = fem.Function(self.temp_V)
        tag.name = 'MeshTAG'
        marker_unique = np.unique(D.facets.values)
        for i in range(len(marker_unique)):
            facet_indices = D.facets.find(marker_unique[i])   # numpy array of facet ids
            dofs = fem.locate_dofs_topological(self.temp_V, 1, facet_indices)
            tag.x.array[dofs] = marker_unique[i]
            tag.x.scatter_forward()
        
        with XDMFFile(MPI.COMM_WORLD, "%s.xdmf"%file_name2, "w") as ufile_xdmf:
        
            coord = D.mesh.geometry.x.copy()
            D.mesh.geometry.x[:] *= sc.L/1e3
            ufile_xdmf.write_mesh(D.mesh)
            ufile_xdmf.write_function(tag)
            D.mesh.geometry.x[:] = coord


        if ctrl.steady_state == 0:
            # transient: append to ongoing XDMF with time
            # write each field at this physical_time
            # ...same for PL2, rho2, Cp2, k2, kappa2, e_T, eta2, flux
            self.xdmf_main.write_function(dT,           time)
            self.xdmf_main.write_function(u_T,          time)
            self.xdmf_main.write_function(p2,           time)
            self.xdmf_main.write_function(T2,           time)
            self.xdmf_main.write_function(PL2,          time)
            self.xdmf_main.write_function(rho2,         time)
            self.xdmf_main.write_function(Cp2,          time)
            self.xdmf_main.write_function(k2,           time)
            self.xdmf_main.write_function(kappa2,       time)
            self.xdmf_main.write_function(alpha2,       time)
            self.xdmf_main.write_function(e_T,          time)
            self.xdmf_main.write_function(eta2,         time)
            self.xdmf_main.write_function(eta2,         time)
            self.xdmf_main.write_function(shear_heatingF,   time)
            self.xdmf_main.write_function(tag,          time)
        else:
            with XDMFFile(MPI.COMM_WORLD, "%s.xdmf"%file_name, "w") as ufile_xdmf:

                coord = D.mesh.geometry.x.copy()
                D.mesh.geometry.x[:] *= sc.L/1e3
                ufile_xdmf.write_mesh(D.mesh)
                ufile_xdmf.write_function(u_T  )
                ufile_xdmf.write_function(p2   )
                ufile_xdmf.write_function(T2   )
                ufile_xdmf.write_function(PL2  )
                ufile_xdmf.write_function(rho2 )
                ufile_xdmf.write_function(Cp2  )
                ufile_xdmf.write_function(k2   )
                ufile_xdmf.write_function(kappa2)
                ufile_xdmf.write_function(alpha2)
                ufile_xdmf.write_function(e_T  )
                ufile_xdmf.write_function(eta2 )
                ufile_xdmf.write_function(adiabaticH )
                ufile_xdmf.write_function(shear_heatingF)
                ufile_xdmf.write_function(adiabatic_comp)

                ufile_xdmf.write_function(tag )
                D.mesh.geometry.x[:] = coord



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
    from .utils import gather_vector,gather_coordinates
    
    comm = MPI.COMM_WORLD
    # Suppose u is your Function
    # u = fem.Function(V)

    lT = S.T_O.copy()#gather_vector(S.T_O.copy())
    mpi_comm = lT.function_space.mesh.comm
    array = lT.x.array

    # gather solution from all processes on proc 0
    gT = mpi_comm.gather(array, root=0)
    
    XGl = S.T_O.function_space.tabulate_dof_coordinates()#gather_coordinates(S.T_O.function_space)
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
        idx1 = 36-1 
        idy1 = ny-36

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
        van_keken_db.close()

    return 0 