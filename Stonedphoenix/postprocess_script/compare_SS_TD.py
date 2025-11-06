import vtk 
from mpi4py import MPI
import numpy as np
import dolfinx 
import h5py
from dolfinx.io import XDMFFile
import os

def compare_SS_TD(ss_file, td_file, time_td):
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    
    
    f = h5py.File(td_file, 'r')
    field = 'Function/Temperature  [degC]'
    times = list(f[field].keys())
    time_list = [float(s.replace("_", "")) for s in times]
    X = np.array(f['/Mesh/mesh/geometry'])
    x_min = np.min(X[:,0])
    x_max = np.max(X[:,0])
    y_min = np.min(X[:,1])
    y_max = np.max(X[:,1])

    # Load steady state file    
    for i in times:
        
        T = np.array(f[field+'/'+i])
        T_i = griddata(X, T, (Xi, Yi), method='linear', fill_value=np.nan)
        
    
    
    
    
    
    return 0 


if __name__ == "__main__":
    td_file = '/Users/wlnw570/Work/Leeds/Output/Stonedphoenix/curved/case_Hobson_time_dependent_experiment/timeseries_all.h5'
    ss_file = '/path/to/time_dependent_output.vtu'
    time_td = 10.0  # Time in Myr to compare
    
    compare_SS_TD(ss_file, td_file, time_td)
    
