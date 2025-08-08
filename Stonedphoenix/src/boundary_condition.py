import sys, os
sys.path.insert(0, "/Users/wlnw570/Work/Leeds/Fenics_tutorial/Stonedphoenix")
import gmsh 
import meshio
from mpi4py                          import MPI
from petsc4py                        import PETSc
from dolfinx                         import mesh, fem, io, nls, log
from dolfinx.fem.petsc               import NonlinearProblem
from dolfinx.nls.petsc               import NewtonSolver
from dolfinx.io                      import XDMFFile, gmshio
from ufl                             import exp, conditional, eq, as_ufl
from src.create_mesh                 import Mesh
from src.numerical_control           import NumericalControls, ctrl_LHS
from src.numerical_control           import IOControls 
from src.solution                    import Solution 
from src.compute_material_property   import density_FX
from src.compute_material_property   import heat_conductivity_FX
from src.compute_material_property   import heat_capacity_FX

import ufl

from   src.scal                      import Scal
from   src.create_mesh               import Mesh
from   src.phase_db                  import PhaseDataBase
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

import matplotlib.pyplot             as plt
import compute_material_property     as cmp 
import src.scal                      as sc_f 
import basix.ufl
import time                          as timing
import scipy.interpolate             as interp 
from scipy.interpolate import griddata
from ufl import FacetNormal, ds, dot, sqrt, as_vector
from create_mesh import dict_tag_lines 



"""
-> a) Find nodes per each boundary condition 
-> b) Assign type per each boundary condition 
-> c) Assign value per each boundary condition


"""




def find_coordinate_facet(M, physical_line):
    " Simple function to extract the coordinate of the given physical line: argument: M mesh, physical line the tag of the line"
    fdim     = M.mesh.topology.dim - 1    
    facets   = M.mesh_Ftag.find(physical_line)
    facet_vertices = M.mesh.topology.connectivity(1, 0)  # facet to vertex connectivity
    facet_vertices = np.unique(np.concatenate([facet_vertices.links(facet) for facet in facets]))
    # Extract coordinates
    coords = M.mesh.geometry.x[facet_vertices]
    
    return coords    


#---
class bc_energy():
    
    
    def __init__(self, M, X, ctrl, lhs,v):
         
        self.top       = self.create_bc_temp( M, X, ctrl, lhs, dict_tag_lines['Top'],       v)
        self.right_B   = self.create_bc_temp( M, X, ctrl, lhs, dict_tag_lines['Right_wed'], v)
        self.right_C   = self.create_bc_temp( M, X, ctrl, lhs, dict_tag_lines['Right_lit'], v)
        self.bottom_B  = self.create_bc_temp( M, X, ctrl, lhs, dict_tag_lines['Bottom_wed'],v)
        self.bottom_A  = self.create_bc_temp( M, X, ctrl, lhs, dict_tag_lines['Bottom_sla'],v)
        self.left      = self.create_bc_temp( M, X, ctrl, lhs, dict_tag_lines['Left_inlet'],v)
    
    def create_bc_temp(self,M,X,ctrl,lhs,physical_line,v):
    
        fdim     = M.mesh.topology.dim - 1    
        facets   = M.mesh_Ftag.find(physical_line)
        dofs     = fem.locate_dofs_topological(X, M.mesh.topology.dim-1, facets)
        if physical_line == 1: 
            # -> Probably I need to use some parallel shit here 
            bc = fem.dirichletbc(ctrl.Ttop, dofs, X) 
        elif physical_line == 3 or physical_line == 2: 
            bc = self.right_boundary_thermal(fdim, facets, dofs, M, X,v,ctrl,physical_line)
        
        elif physical_line == 4 or physical_line == 5: 
            # Do nothing bc -> tough I would like to have a similar approach for the right bc for safety reason
            # or U if it's already a UFL expression
            v_vel  = v.sub(1) # index 1 = y-direction (2D)
        
            vel_T  = fem.Function(X)
            vel_T.interpolate(v_vel)
            vel_bc = vel_T.x.array[dofs]
            ind_z = np.where(vel_bc >= 0.0)
            dofs_vel = dofs[ind_z[0]]                
            bc = fem.dirichletbc(ctrl.Tmax, dofs_vel,X)

        elif physical_line == 7: 
            bc = self.left_thermal_boundary(fdim, facets, dofs, M, X,ctrl,lhs)
        
        
        return bc 
    
    def right_boundary_thermal(self,fdim, facets, dofs, M, X, v, ctrl, domain):
        # - > set the lithosphere -> then set the area in which exist a negative velocity field along x
        # Extract coordinate of the dof
        if domain == 2:
            cd_dof = X.tabulate_dof_coordinates()
            cd_dof_b = cd_dof[dofs]
        

        
            T_gr = (-M.g_input.lt_d-0)/(ctrl.Tmax-ctrl.Ttop)
            T_gr = T_gr**(-1) 
        
            bc_fun = fem.Function(X)
            bc_fun.x.array[dofs] = ctrl.Ttop + T_gr * cd_dof[dofs,1]
            bc_fun.x.scatter_forward()
        
            bc = fem.dirichletbc(bc_fun, dofs)
        else: 
         # or U if it's already a UFL expression
            h_vel  = v.sub(0) # index 1 = y-direction (2D)
        
            vel_T  = fem.Function(X)
            vel_T.interpolate(h_vel)
            vel_bc = vel_T.x.array[dofs]
            ind_z = np.where((vel_bc <= 0.0))
            dofs_vel = dofs[ind_z[0]]        
        
            bc = fem.dirichletbc(ctrl.Tmax, dofs_vel,X)

        return bc 

    def left_thermal_boundary(self,fdim, facets, dofs, M, X, ctrl, lhs):
        
        # Create suitable function space for the problem
        T_bc_L = fem.Function(X)
        # Extract z and lhs 
        z   = - lhs.z
        LHS = lhs.LHS 
        # Extract coordinate dofs
        cd_dof = X.tabulate_dof_coordinates()
        # Interpolate temperature field: 
        T_bc_L.x.array[:] = griddata(z, LHS, cd_dof[:,1], method='nearest')
        T_bc_L.x.scatter_forward()
        bc = fem.dirichletbc(T_bc_L, dofs)
        return bc 
    
    
    

                        
        
        
        

        
        
        

    





def unit_test(): 
    from create_mesh import unit_test_mesh
    from phase_db import _generate_phase
    from phase_db import PhaseDataBase 
    from thermal_structure_ocean import compute_initial_LHS


        # Create scal 
    sc = Scal(L=660e3,Temp = 1350,eta = 1e21, stress = 1e9)
    
    ioctrl = IOControls(test_name = 'Debug_test',
                        path_save = '../Debug_mesh',
                        sname = 'Experimental')
    ioctrl.generate_io()
    
    # Create mesh 
    M = unit_test_mesh(ioctrl, sc)
        
    print('mesh done')
    
    # Create initial temperature field
    
    M.T_i.x.array[:] = 1300/sc.Temp
    
    # Create Controls 
    
    ctrl = NumericalControls()
    
    ctrl = sc_f._scaling_control_parameters(ctrl, sc)
    

    
    pdb = PhaseDataBase(8)
    # Slab
    
    pdb = _generate_phase(pdb, 1, rho0 = 3300 , option_rho = 2, option_rheology = 0, option_k = 3, option_Cp = 3, eta=1e22, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    # Oceanic Crust
    
    pdb = _generate_phase(pdb, 2, rho0 = 2900 , option_rho = 2, option_rheology = 0, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    # Wedge
    
    pdb = _generate_phase(pdb, 3, rho0 = 3300 , option_rho = 2, option_rheology = 3, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    # 
    
    pdb = _generate_phase(pdb, 4, rho0 = 3250 , option_rho = 2, option_rheology = 0, option_k = 3, option_Cp = 3, eta=1e23, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    #
    
    pdb = _generate_phase(pdb, 5, rho0 = 2800 , option_rho = 2, option_rheology = 0, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    #
    
    pdb = _generate_phase(pdb, 6, rho0 = 2700 , option_rho = 2, option_rheology = 0, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    #
    
    pdb = _generate_phase(pdb, 7, rho0 = 3250 , option_rho = 2, option_rheology = 3, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')
    #
    
    pdb = _generate_phase(pdb, 8, rho0 = 3250 , option_rho = 2, option_rheology = 3, option_k = 3, option_Cp = 3, eta=1e21, name_diffusion='Van_Keken_diff', name_dislocation='Van_Keken_disl')

    pdb = sc_f._scaling_material_properties(pdb,sc)
    
    timeA = timing.time()
    lhs_ctrl = ctrl_LHS()

    lhs_ctrl = sc_f._scale_parameters(lhs_ctrl, sc)
    
    lhs_ctrl = compute_initial_LHS(ctrl,lhs_ctrl, sc, pdb)
    timeB = timing.time()
    
    
    # Create the meshesh for the testing
    
    
    X = M.PT 
    
    element_u = basix.ufl.element("Lagrange", "triangle", 2, shape=(2,))
    V = fem.functionspace(M.mesh, element_u)
    v = fem.Function(V) 
    v.x.array[:] = 0.0 


    BC_E = bc_energy(M, X, ctrl, lhs_ctrl,v)
    BC_S = bc_stokes(M,V,ctrl)
    

    pass 



if __name__ == '__main__':
    
    unit_test()
    
    
    