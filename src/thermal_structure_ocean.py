# calculate the thermal structure of the ocean according to Richards et al., 2018 JGR: Solid Earth 
# we use a time- and space-centered Crank-Nicholson finite-difference scheme with a predictor-corrector step (Press et al., 1992)

# import all the constants and defined model setup parameters 
import sys,os,fnmatch

# modules
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import time as timing
import scipy.linalg as la 
import scipy.sparse as sps
import scipy.sparse.linalg.dsolve as linsolve
from .compute_material_property import heat_capacity,heat_conductivity,density
from numba import njit

import matplotlib.pyplot as plt
from matplotlib import rcParams
from .utils import timing_function, print_ph
from .compute_material_property import compute_thermal_properties

start       = timing.time()
zeros       = np.zeros
power       = np.power
exp         = np.exp
transpose   = np.transpose 
array       = np.array
full        = np.full
all         = np.all
diagonal    = np.diagonal
solve_banded= la.solve_banded

@njit
def _compute_lithostatic_pressure(nz,ph,g,dz,T,pdb):
    #compute lithostatic pressure:
    lit_p_o =  zeros((nz))
    lit_p   = zeros((nz))
    res = 1.0
    while res>1e-6:
        for i in range(0,nz):
            rho = density(pdb,T[i],lit_p_o[i],ph[i])
            rhog = rho*g
            if i == 0:
                lit_p[i] = 0.0
            else: 
                lit_p[i] = lit_p[i-1]+rhog*dz 
        
        res =  np.linalg.norm(lit_p-lit_p_o,2)/np.linalg.norm(lit_p+lit_p_o,2)
        lit_p_o[:]=lit_p 
    return lit_p
    
        
# we will solve the system Mx = D with
# x = the temperature vector (our unknowns)
# M = our matrix of coefficients in front of the temperature (known)
# D = the right hand side vector (known)

# first, we need to build M and D
# M and D are different in the predictor and corrector step 
#-----------------------------------------------------------------------------------------
# Melting parametrisation
def compute_initial_geotherm(Tp,z,sc,option_melt = 1 ):
    
    if option_melt == 1: 
        
        from Melting_parametrisation import lithologies,compute_Fraction,runge_kutta_algori,SL,dSL,rcpx,compute_T_cpx
        
        TS       = lambda  P :SL(1085.7, 132.9,-5.1,P)

        dTS      = lambda  P :dSL(132.9,-5.1,P)

        TLL      = lambda  P :SL(1475,80,-3.2,P)

        dTLL     = lambda P  :dSL(80,-3.2,P)

        TL       = lambda  P :SL(1780,45.0,-2.0,P)

        dTL      = lambda  P :dSL(45.0,-2.0,P)

        Mcpx     = lambda  fr,P :rcpx(0.5,0.08,fr,P)

        TCpx     = lambda  fr,P :compute_T_cpx(Mcpx(fr,P),TS(P),TLL(P))

        #Ts = None,Tcpx=None,TLl=None,TL = None,MCpx = None,dTs = None, dTL = None, dTLl = None
        lhz = lithologies(Ts=TS,Tcpx=TCpx,TLl=TLL,TL=TL,MCpx=Mcpx,dTs=dTS,dTL=dTL,dTLl=dTLL)

        P = (z*sc.L * 9.81 * lhz.rho) 
        
        T = np.zeros(len(P))
        
        P = np.flip(P)
        
        Tp = Tp*sc.Temp
        T_start = (Tp) + 18/1e9*P[0]
        
        T[0] = T_start
        fig = plt.figure()
        ax = fig.gca()

        for i in range(len(P)-1):
            dP = P[i+1]- P[i]
            F0  = compute_Fraction(T[i],TL(P[i]),TLL(P[i]),TS(P[i]),TCpx(lhz.fr,P[i]),Mcpx(lhz.fr,P[i]))  
            dT = runge_kutta_algori(lhz,P[i],T[i],dP,F0,None,Tp)
            T[i+1] = T[i] + dT 
            if F0>0.0 and F0 <= lhz.Mcpx(lhz.fr,P[i]):
                c = 'r'
                alpha = 0.8 
            elif F0 > lhz.Mcpx(lhz.fr,P[i]):
                c = 'forestgreen'
                alpha = 0.9
            else:
                c='k'
                alpha = 0.3


            ax.scatter(T[i]-273.15,P[i]/1e9,c=c,s=0.5,alpha = alpha)   

        ax.invert_yaxis()

        ax.plot(lhz.Ts(P) - 273.15 , P/1e9 ,linestyle='-.', c = 'k'          , linewidth = 0.8)
        ax.plot(lhz.TLL(P) - 273.15, P/1e9 ,linestyle='--', c = 'forestgreen', linewidth = 0.8)
        ax.plot(lhz.TL(P)  - 273.15, P/1e9 ,linestyle='-.', c = 'firebrick'  , linewidth = 0.8)
        ax.plot(lhz.Tcpx(lhz.fr,P)- 273.15, P/1e9 ,linestyle='--', c = 'b'          , linewidth = 0.8)
        
        Told = np.flip(T)/sc.Temp


    else: 
        
        Told = np.ones(len(z))*Tp
     
    
    
    
    
    return Told 

#-----------------------------------------------------------------------------------------
@njit
def _compute_Cp_k_rho(ph,pdb,Cp,k,rho,T,p,ind_z):
    it = len(T)

    for i in range(it):
        Cp[i], rho[i], k[i] = compute_thermal_properties(pdb,T[i],p[i],ph[i])
        

    return Cp,k,rho
#-----------------------------------------------------------------------------------------
@njit
def build_coefficient_matrix(A,
                             B,
                             D,
                             Q,
                             M,
                             pdb,
                             ph,
                             ctrl,
                             lhs,
                             TO,
                             TG,
                             TPr,
                             k_m,
                             density_m,
                             heat_capacity_m,
                             step,
                             lit_p,
                             sc,
                             ind_z,
                             Ttop,
                             Tmax):

    nz         = lhs.nz 
    dt         = lhs.dt 
    dz         = lhs.dz
    

    
    if step == 0:

        # predictor step

        # m = n

        heat_capacity_m,k_m,density_m=_compute_Cp_k_rho(ph,pdb,heat_capacity_m,k_m,density_m,TG,lit_p,ind_z)

        dz_m               = full((nz), dz) # current assumption: incompressible

    elif step == 1: 

        # corrector step 

        # m = n+1/2

        Cp_m0,k_m0,rho_m0=_compute_Cp_k_rho(ph,pdb,heat_capacity_m,k_m,density_m,TG,lit_p,ind_z)

        Cp_m1,k_m1,rho_m1=_compute_Cp_k_rho(ph,pdb,heat_capacity_m,k_m,density_m,TG,lit_p,ind_z)

        heat_capacity_m = (Cp_m1+Cp_m0)/2

        k_m              = (k_m0+k_m1)/2

        density_m        = (rho_m0+rho_m1)/2

        dz_m           = (full((nz), dz) + full((nz), dz) ) / 2.  # current assumption: incompressible


    for i in range(0,nz):

        if i == 0: 

            start_loop = i

            end_loop   = i+1

        elif i == nz-1:

            start_loop = i-1

            end_loop   = i  

        else: 

            start_loop = i-1

            end_loop   = i+1  


        for j in range(start_loop,end_loop+1): 

        

            if j == 0:

                # calculate A  

                A[j] = (dt / ( density_m[j] * heat_capacity_m[j] * ( dz_m[j] + dz_m[j] ) ))

            elif j > 0:

                # calculate A 

                A[j] = (dt / ( density_m[j] * heat_capacity_m[j] * ( dz_m[j] + dz_m[j-1] ) ))  


            # ========== BUILD M ========== - coefficient matrix 

            # ========== boundary conditions ==========

            if (i == 0 and j == 0): 

                # boundary condition at the top 

                M[i,j] = 1. 

                D[j]   = Ttop 

            elif (i == nz-1 and j == nz-1):

                # boundary condition at the bottom 

                M[i,j] = 1. 

                D[j]   = Tmax 

            else:

                # ========== BUILD M ========== - coefficient matrix 

                if i - j == 1 and j < nz - 2:

                    # T_j+1 

                    if j > 0:

                        M[i,j] = -A[j] * ( (k_m[j] + k_m[j+1] ) / 2. ) / dz_m[j]

                    elif j == 0:

                        M[i,j] = -A[j] * ( (k_m[j] + k_m[j] ) / 2. ) / dz_m[j]

                elif i == j:

                    # diagonal: T_j

                    M[i,j] = 1. + A[j] * ( ( (k_m[j] + k_m[j+1] ) / 2. ) / dz_m[j] + ( (k_m[j] + k_m[j-1] ) / 2. ) / dz_m[j-1])

                elif i - j == -1 and j > 1 and j < nz:

                    # T_j-1

                    if j < nz - 1:

                        M[i,j] = -A[j] * ( (k_m[j] + k_m[j-1] ) / 2. ) / dz_m[j-1]

                    elif j == nz - 1:

                        M[i,j] = -A[j] * ( (k_m[j] + k_m[j] ) / 2. ) / dz_m[j]


                # ========== BUILD D ========== - right hand side vector 

                # D consists of multiple components

                # we say D = T + A * Q + B

                if j > 0 and j < nz-1:

                    Q[j] = ( ( (k_m[j] + k_m[j+1] ) / 2. ) / dz_m[j] )*TO[j+1] - (( ( (k_m[j] + k_m[j+1] ) / 2. ) / dz_m[j] ) + ( ( (k_m[j] + k_m[j-1] ) / 2. ) / dz_m[j-1] )) * TO[j] + ( ( (k_m[j] + k_m[j-1] ) / 2. ) / dz_m[j-1] )*TO[j-1]


                    # B - correction that represents the second term on the right-hand side on the equation 

                    rho_A =  density(pdb,TO[j],lit_p[j],ph[j])
                    Cp_A  = heat_capacity(pdb,TO[j],ph[j])

           
                    if step == 0:
                        
                        rho_B = density(pdb,TO[j],lit_p[j],ph[j])
                        Cp_B = heat_capacity(pdb,TO[j],ph[j])

                        # B - predictor step 
                        B[j] = -TO[j] * ( rho_A * Cp_A - rho_B * Cp_B) / (rho_A * Cp_A)

                    elif step == 1: 
                        rho_B = density(pdb,TPr[j],lit_p[j],ph[j])
                        Cp_B = heat_capacity(pdb,TPr[j],ph[j])

                        # B - corrector step 

                        B[j] = - ((TPr[j] + TO[j]) * ( rho_B*Cp_B - rho_A*Cp_A ) / ( rho_B*Cp_A + rho_B*Cp_A))


                    D[j] = TO[j] + A[j] * Q[j] + B[j]
    return M,D

#-----------------------------------------------------------------------------------------
#@njit 
def compute_ocean_plate_temperature(ctrl,lhs,scal,pdb):
    
    

    # Spell out the structure 
    dz          = lhs.dz # m 
    dt          = lhs.dt # year
    end_time    = lhs.end_time    # million years 
    nz          = lhs.nz                 # size of the spatial array (in km; max depth, assuming dz = 1km) 
    depth_melt = lhs.depth_melt
    alpha_g    = lhs.alpha_g
    g          = ctrl.g
    Tmax       = ctrl.Tmax
    Ttop       = ctrl.Ttop
    van_keken_option = lhs.van_keken

    
    option_1D_solve = lhs.option_1D_solve 

    nt = int(end_time / dt + 1)

    t = zeros((1,nt))


    # ========== initial conditions ========== 



    if lhs.van_keken == 1: 
        from scipy import special
        Cp    = 1250/scal.Cp
        k     = 3.0/scal.k
        rho   = 3300/scal.rho
        kappa = k/rho/Cp 
        t     = 50 * scal.scale_Myr2sec/scal.T
        T_lhs = Ttop+(Tmax-Ttop) * special.erf(z /2 /np.sqrt(t * kappa))
        lhs.z[:] = -z[:] 
        lhs.LHS[:] = T_lhs[:]

        return lhs,[],[]


    

    z    = np.arange(0,nz*dz,dz)
    ph   = np.zeros([nz],dtype = np.int32) # I assume that everything is mantle 
    ph[z<6000/scal.L] = np.int32(1)
    
    Told = compute_initial_geotherm(ctrl.Tmax,z,scal,ctrl.adiabatic_heating)
    
    

    
    lit_p = _compute_lithostatic_pressure(nz,ph,g,dz,Told,pdb)
    


    temperature       = zeros([nt,nz],dtype=np.float64)
    pressure          = zeros([nt,nz],dtype=np.float64)
    conductivity      = zeros([nt,nz],dtype=np.float64)
    capacity          = zeros([nt,nz],dtype=np.float64)
    density           = zeros([nt,nz],dtype=np.float64)

    T_1t = zeros([nt,nz],dtype=np.float64)
    temperature[0,:] = Told
    pressure[0,:] = lit_p
    # print(Told)


    k_m = zeros((nz),dtype=np.float64)
    heat_capacity_m = zeros((nz),dtype=np.float64)
    density_m = zeros((nz),dtype=np.float64)
    k_tmp = zeros((nz),dtype=np.float64)
    Cp_tmp = zeros((nz),dtype=np.float64)
    rho_tmp = zeros((nz),dtype=np.float64)

    Ttop = ctrl.Ttop
    Tmax = np.max(Told)

    ind_z_lit = np.where(z>=120e3/scal.L)[0][0] # I force the temperature to be T_max at this dept -> Otherwise I have a temperature jump, and i do not know how much
    for time in range(1,nt):
          
        t[0,time] = t[0,time] + time*dt
        TO = temperature[time-1,:] # temperature at the previous time step
        TG  = TO

        M = zeros((nz, nz),dtype=np.float64)     # pre-allocate M array


        A               = zeros((nz),dtype=np.float64)

        D               = zeros((nz),dtype=np.float64)     # pre-allocate D column vector

        Q               = zeros((nz),dtype=np.float64)

        B               = zeros((nz),dtype=np.float64)

        TPr = np.zeros(nz,dtype=np.float64)

        it = 0 
        res = 1.0
        while res > 1e-6:
            for step in range(2):   


                    M,D = build_coefficient_matrix(A,B,D,Q,M,pdb,ph,ctrl,lhs,TO,TG,TPr,k_m,density_m,heat_capacity_m,step,lit_p,scal,ind_z_lit,Ttop,Tmax)
                    # ========== Solve system of equations using numpy.linalg.solve ==========

                    if option_1D_solve == 1:
                        # brute force solve
                        Tnew = transpose(np.linalg.solve(M, D))    # solve Mx = D for x
                    elif option_1D_solve == 2:    
                        # make ordered banded diagonal matrix MD from M 
                        upper = 1
                        lower = 1
                        n = M.shape[1]
                        assert(all(M.shape ==(n,n)))

                        ab = zeros((2*n-1, n))

                        for i in range(n):
                            ab[i,(n-1)-i:] = diagonal(M,(n-1)-i)

                        for i in range(n-1): 
                            ab[(2*n-2)-i,:i+1] = diagonal(M,i-(n-1))

                        mid_row_inx = int(ab.shape[0]/2)
                        upper_rows = [mid_row_inx - i for i in range(1, upper+1)]
                        upper_rows.reverse()
                        upper_rows.append(mid_row_inx)
                        lower_rows = [mid_row_inx + i for i in range(1, lower+1)]
                        keep_rows = upper_rows+lower_rows
                        ab = ab[keep_rows,:]

                        Tnew = transpose(solve_banded((1,1),ab, D))

                    if step == 0:
                        TPr = Tnew 
                    elif step == 1 and it == 0:
                        T_1t[time,:] = Tnew
            res = np.linalg.norm(Tnew-TG,2)/np.linalg.norm(Tnew+TG,2)
            if np.isnan(res) == True:
                print("NaN detected in the residual")
                sys.exit(1)
            TG = Tnew 
            lit_p = _compute_lithostatic_pressure(nz,ph,g,dz,Told,pdb)
            it += 1

            # end for-loop predictor-corrector step 
        lit_p = _compute_lithostatic_pressure(nz,ph,g,dz,Told,pdb)
        Cp_tmp,k_tmp,rho_tmp = _compute_Cp_k_rho(ph,pdb,Cp_tmp,k_tmp,rho_tmp,Tnew,lit_p,0)
        temperature[time,:] = Tnew
        pressure[time,:]    = lit_p
        capacity[time,:]    = Cp_tmp
        conductivity[time,:] = k_tmp 
        density[time,:]      = rho_tmp
        
    # -> Save the actual LHS 
    # Correction 
    #temperature[:,-z<-120e3/scal.L] = ctrl.Tmax

    # Current age index
    current_age_index = np.where(t[0] >= lhs.c_age_plate)[0][0]
    lhs.LHS[:] = temperature[current_age_index]
    lhs.z[:] = - z

    TTime,ZZ = np.meshgrid(t[0],z)
    TTime = TTime*scal.T/365.25/60/60/24/1e6
    TTime = TTime[:,1::]
    ZZ    = ZZ[:,1::] 
    ZZ    = ZZ*scal.L/1e3
    Tem   = temperature.T[:,1::]*scal.Temp - 273.15
    Cp    = capacity.T[:,1::]*scal.Cp 
    k     = conductivity.T[:,1::]*scal.k 
    rho   = density.T[:,1::]*scal.rho 
    LP    = pressure.T[:,1::]*scal.stress/1e9

    
    


    return lhs, t, temperature


def compute_topography(T,rho):
    
    
    
    
    
    pass
    
    #return H 


@timing_function
def compute_initial_LHS(ctrl,lhs,scal,pdb):
    
    if lhs.van_keken == 0:
        lhs,t, temperature = compute_ocean_plate_temperature(ctrl,lhs,scal,pdb)
    else:
        lhs,_,_ = compute_ocean_plate_temperature(ctrl,lhs,scal,pdb)
        return lhs
    
    
    
    from scipy.interpolate import RegularGridInterpolator
 
    t_re = np.linspace(lhs.c_age_var[0],lhs.c_age_var[1], num = lhs.LHS_var.shape[1])
    T,Z  = np.meshgrid(t_re,lhs.z,indexing='ij')
    TT,ZZ = np.meshgrid(t,lhs.z,indexing='ij')
    interp_func = RegularGridInterpolator((t[0], lhs.z), temperature)
    points_coarse = np.column_stack((T.ravel(), Z.ravel()))
    lhs.LHS_var = interp_func(points_coarse).reshape(len(t_re), len(lhs.z))
    lhs.t_res_vec = t_re 
        
    
    return lhs
#-----------------------------------------------------------------------------------------



    #print("time: %.3f s" % (timing.time() - start))
    # print(Tnew)

    # ========== save thermal structure ==========
    #if os.path.isdir('databases') == False:
    #    os.makedirs('databases')

    #fname = 'databases/slab_k{0}_Cp{1}_rho{2}'.format(option_k,option_Cp,option_rho)
    #np.savez('%s.npz'%fname,temperature=temperature,t=t,z=z)
    
    
    """
        visualisation_1D = 1
    if visualisation_1D == 1:

        X, Y  = np.meshgrid(t*scal.T/1e6/364.25/24/60/60, z*scal.L/1e3)     # make a mesh of X and Y for easy plotting in the right units (i.e., t in Myr and z in km) 
        Zaxis = transpose(temperature-273.15)                   # convert temperature back into C for easy plotting
        Zaxis2 = transpose(pressure/1e9)
        #matplotlib inline
        fn = '%s_T.png'%fname
        fn2 = '%s_P.png'%fname

        
        fg = plt.figure()
        # Initialize plot objects
        rcParams['figure.figsize'] = 10, 5 # sets plot size
        ax = fg.gca()

        # define contours
        levels = array([200., 400., 600., 800., 1000., 1200.])

        cpf = ax.contourf(X,-Y,Zaxis, len(levels), cmap='cmc.lipari')

        line_colors = ['black' for l in cpf.levels]

        # Generate a contour plot
        cp = ax.contour(X, -Y, Zaxis, levels=levels, colors=line_colors)
        #plt.colorbar(orientation='vertical',label=r'Temperature, $[^\circ]C$') 
        ax.clabel(cp, fontsize=8, colors=line_colors)
        ax.set_xlabel('Age [Myr]')
        ax.set_ylabel('Depth, [km]')
        fg.savefig(fn,dpi=600,transparent=False)
        fg = plt.figure()
        # Initialize plot objects
        rcParams['figure.figsize'] = 10, 5 # sets plot size
        ax = fg.gca()

        # define contours
        levels = np.linspace(0,3.5,20)

        cpf = ax.contourf(X,-Y,Zaxis2, len(levels), cmap='cmc.lipari')

        line_colors = ['black' for l in cpf.levels]

        # Generate a contour plot
        #plt.colorbar(orientation='vertical',label=r'Lithostatic Pressure, [GPa]') 
        #ax.clabel(cp, fontsize=8, colors=line_colors)
        ax.set_xlabel('Age, [Myr]')
        ax.set_ylabel('Depth, [km]')
        fg.savefig(fn2,dpi=600,transparent=False)

    import cmcrameri as cmc
    
    
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["mathtext.rm"] = "serif"
    
    
    fg = plt.figure(figsize=(10,6))
    ax0 = fg.gca()
    a = ax0.contourf(TTime,-ZZ,Tem, levels=[0,100,200,300,500,600,700,800,900,1000,1100,1200,1300,1400], cmap='cmc.lipari')
    b = ax0.contour(TTime, -ZZ, Tem,levels=[100,200,300,500,600,700,800,900,1000,1100,1200,1300],colors='white')
    
    ax0.set_xlim(0,150)
    ax0.set_ylim(-150,0.0)

    plt.colorbar(a, label=r'T, $[^{\circ}C]$', location='bottom')    
    plt.ylabel('Depth [km]')
    plt.xlabel('Time  [Myr]')
    plt.title('Plate model')
    plt.show()
    plt.savefig('Temp.png')

    
    fg = plt.figure(figsize=(10,6))
    ax0 = fg.gca()
    a = ax0.contourf(TTime,-ZZ,Cp, levels=20, cmap='cmc.lipari')
    plt.colorbar(a, label=r'Cp, $[J/K/kg]$', location='bottom')    
    plt.ylabel('Depth [km]')
    plt.xlabel('Time  [Myr]')
    plt.title('Plate model')
    plt.show()
    fg.savefig('Cp.png')

    
    fg = plt.figure(figsize=(10,6))
    ax0 = fg.gca()
    a = ax0.contourf(TTime,-ZZ,k,  levels=40, cmap='cmc.nuuk')


    plt.colorbar(a, label=r'k, $[W/K/m]$', location='bottom')    
    plt.ylabel('Depth [km]')
    plt.xlabel('Time  [Myr]')
    plt.title('Plate model')
    plt.show()
    fg.savefig('Condu.png')

    
    fg = plt.figure(figsize=(10,6))
    ax0 = fg.gca()
    a = ax0.contourf(TTime,-ZZ,rho, levels=10, cmap='cmc.lipari')

    plt.colorbar(a, label=r'$\rho$, $[kg/m^3]$', location='bottom')    
    plt.ylabel('Depth [km]')
    plt.xlabel('Time  [Myr]')
    plt.title('Plate model')
    plt.show()
    fg.savefig('rho.png')
    fg = plt.figure(figsize=(10,6))
    ax0 = fg.gca()
    a = ax0.contourf(TTime,-ZZ,LP, levels=10, cmap='cmc.lipari')

    plt.colorbar(a, label=r'$P$, $[GPa]$', location='bottom')    
    plt.ylabel('Depth [km]')
    plt.xlabel('Time  [Myr]')
    plt.title('Plate model')
    plt.show()
    fg.savefig('Pressure.png')


    """