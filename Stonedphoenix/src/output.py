"""_summary_
    General output folder-> where I write the functions required to print the file and databases
    
"""
    
from dolfinx import fem
from mpi4py import MPI
import ufl
from .compute_material_property import density_FX,heat_capacity_FX,heat_conductivity_FX
from .compute_material_property import compute_viscosity_FX
from .utils import compute_strain_rate,compute_eII
import os
from dolfinx.io import XDMFFile
import numpy as np
from petsc4py import PETSc


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
        
        from .compute_material_property import evaluate_material_property  
        import os
        from dolfinx.io import XDMFFile
        import basix  

        if ctrl.steady_state == 1:
            file_name = os.path.join(self.pt_save,'Steady_stateit_%04d'%(it_outer))



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

        if ctrl.steady_state == 0:
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


        # density 
        rho = density_FX(pdb,S.T_O,S.PL,D.phase,D)
        rho2 = evaluate_material_property(rho, self.temp_V)
        rho2.name = "Density  [kg/m3]"
        rho2.x.array[:] = rho2.x.array[:]*sc.rho
        rho2.x.scatter_forward()

        # Cp 
        Cp = heat_capacity_FX(pdb,S.T_O,D.phase,D)
        Cp2 = evaluate_material_property(Cp,self.temp_V)
        Cp2.name = "Cp  [J/kg]"
        Cp2.x.array[:] = Cp2.x.array[:]*sc.Cp
        Cp2.x.scatter_forward()

        # k 
        k = heat_conductivity_FX(pdb,S.T_O,S.PL,D.phase,D)
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
        eta = compute_viscosity_FX(e_T,S.T_O,S.PL,pdb,D.phase,D,sc)
        eta2 = evaluate_material_property(eta,self.temp_V)
        eta2.name = "eta  [Pa s]"
        eta2.x.array[:] = np.abs(eta2.x.array[:])*sc.eta
        eta2.x.scatter_forward()

        # heat flux 
        q_expr = - (k2 * ufl.grad(T2)*1/sc.L)  
        flux   = fem.Function(self.vel_V)
        flux.name ='q  [W/m2]'
        v = ufl.TestFunction(self.vel_V)
        w = ufl.TrialFunction(self.vel_V)
        a = ufl.inner(w,v) * ufl.dx
        L = ufl.inner(q_expr,v) * ufl.dx
        A = fem.petsc.assemble_matrix(fem.form(a))
        A.assemble()
        b = fem.petsc.assemble_vector(fem.form(L))

        solver = PETSc.KSP().create(D.mesh.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.CG)
        solver.getPC().setType(PETSc.PC.Type.JACOBI)
        solver.setFromOptions()
        solver.solve(b, flux.x.petsc_vec)
        flux.x.scatter_forward()


        if ctrl.steady_state == 0:
            # transient: append to ongoing XDMF with time
            # write each field at this physical_time
            # ...same for PL2, rho2, Cp2, k2, kappa2, e_T, eta2, flux
            self.xdmf_main.write_function(dT,     time)
            self.xdmf_main.write_function(u_T,    time)
            self.xdmf_main.write_function(p2,     time)
            self.xdmf_main.write_function(T2,     time)
            self.xdmf_main.write_function(PL2,    time)
            self.xdmf_main.write_function(rho2,   time)
            self.xdmf_main.write_function(Cp2,    time)
            self.xdmf_main.write_function(k2,     time)
            self.xdmf_main.write_function(kappa2, time)
            self.xdmf_main.write_function(e_T,    time)
            self.xdmf_main.write_function(eta2,   time)
            self.xdmf_main.write_function(flux,   time)
        else:
            with XDMFFile(MPI.COMM_WORLD, "%s.xdmf"%file_name, "w") as ufile_xdmf:

                coord = D.mesh.geometry.x.copy()
                D.mesh.geometry.x[:] *= sc.L/1e3
                ufile_xdmf.write_mesh(mesh)
                ufile_xdmf.write_function(u_T)
                ufile_xdmf.write_function(p2)
                ufile_xdmf.write_function(T2)
                ufile_xdmf.write_function(PL2)
                ufile_xdmf.write_function(rho2)
                ufile_xdmf.write_function(Cp2)
                ufile_xdmf.write_function(k2)
                ufile_xdmf.write_function(kappa2)
                ufile_xdmf.write_function(e_T)
                ufile_xdmf.write_function(eta2)
                ufile_xdmf.write_function(flux)
                D.mesh.geometry.x[:] = coord




        return 0
    
    def close(self):
    
        if self.xdmf_main is not None:
            self.xdmf_main.close()