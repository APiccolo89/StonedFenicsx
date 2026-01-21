# calculate the thermal structure of the ocean according to Richards et al., 2018 JGR: Solid Earth 
# we use a time- and space-centered Crank-Nicholson finite-difference scheme with a predictor-corrector step (Press et al., 1992)

# import all the constants and defined model setup parameters 

# modules
from .package_import import *
from .utils import evaluate_material_property
# ---------------------------------------------------------------------------------
@dataclass
class Functions_material_properties_global():
    # Heat Conductivity properties
    k0 : fem.Function = None 
    fr : fem.Function = None
    k_a: fem.Function = None
    k_b: fem.Function = None
    k_c: fem.Function = None
    k_d: fem.Function = None
    k_e: fem.Function = None
    k_f: fem.Function = None
    # Heat Capacity properties
    C0 : fem.Function = None
    C1 : fem.Function = None
    C2 : fem.Function = None
    C3 : fem.Function = None
    C4 : fem.Function = None
    C5 : fem.Function = None
    # Density properties
    rho0    : fem.Function = None
    alpha0  : fem.Function = None
    alpha1  : fem.Function = None
    alpha2  : fem.Function = None
    Kb      : fem.Function = None
    option_rho : fem.Function = None
    # Radiogenic heating properties
    radio   : fem.Function = None
    # Other properties
    Tref    : float = 0.0
    R       : float = 8.3145
    A       : float = 0.0
    B       : float = 0.0
    T_A     : float = 0.0
    x_A     : float = 0.0
    T_B     : float = 0.0
    x_B     : float = 0.0
# ---------------------------------------------------------------------------------
def populate_material_properties_thermal(CP,pdb,phase):
    """
    Initialize all material properties as FEniCS functions.
    """
    ph = np.int32(phase.x.array)
    P0 = phase.function_space
    
    # Heat Conductivity properties
    CP.k0 = fem.Function(P0)  ; CP.k0.x.array[:]    =  pdb.k0[ph]
    CP.fr = fem.Function(P0)  ; CP.fr.x.array[:]     =  pdb.radio_flag[ph]
    CP.k_a= fem.Function(P0)  ; CP.k_a.x.array[:]    = pdb.k_a[ph]
    CP.k_b= fem.Function(P0)  ; CP.k_b.x.array[:]    = pdb.k_b[ph]
    CP.k_c= fem.Function(P0)  ; CP.k_c.x.array[:]    = pdb.k_c[ph]
    CP.k_d= fem.Function(P0)  ; CP.k_d.x.array[:]    = pdb.k_d[ph]
    CP.k_e= fem.Function(P0)  ; CP.k_e.x.array[:]    = pdb.k_e[ph]
    CP.k_f= fem.Function(P0)  ; CP.k_f.x.array[:]    = pdb.k_f[ph]
    # Heat Capacity properties
    CP.C0  = fem.Function(P0) ; CP.C0.x.array[:]     = pdb.C0[ph]
    CP.C1 = fem.Function(P0)  ; CP.C1.x.array[:]     = pdb.C1[ph]
    CP.C2 = fem.Function(P0)  ; CP.C2.x.array[:]     = pdb.C2[ph]
    CP.C3 = fem.Function(P0)  ; CP.C3.x.array[:]     = pdb.C3[ph]
    CP.C4 = fem.Function(P0)  ; CP.C4.x.array[:]     = pdb.C4[ph]
    CP.C5 = fem.Function(P0)  ; CP.C5.x.array[:]     = pdb.C5[ph]
    # Density properties
    CP.rho0    = fem.Function(P0)     ; CP.rho0.x.array[:]    = pdb.rho0[ph]
    CP.alpha0  = fem.Function(P0)     ; CP.alpha0.x.array[:]  = pdb.alpha0[ph]
    CP.alpha1  = fem.Function(P0)     ; CP.alpha1.x.array[:]  = pdb.alpha1[ph]
    CP.alpha2  = fem.Function(P0)     ; CP.alpha2.x.array[:]  = pdb.alpha2[ph]
    CP.Kb      = fem.Function(P0)     ; CP.Kb.x.array[:]      = pdb.Kb[ph]
    CP.option_rho = fem.Function(P0)  ; CP.option_rho.x.array[:] = pdb.option_rho[ph]
    
    CP.radio   = fem.Function(P0)   ; CP.radio.x.array[:]     = pdb.radio[ph]
    CP.radio.x.scatter_forward()
    CP.Tref    = pdb.Tref
    CP.R       = pdb.R
    CP.A       = pdb.A
    CP.B       = pdb.B
    CP.T_A     = pdb.T_A
    CP.x_A     = pdb.x_A
    CP.T_B     = pdb.T_B
    CP.x_B     = pdb.x_B
    
    return CP

        
        
        
        
        
@dataclass
class Functions_material_rheology():
    Bdif    : fem.Function = None
    Bdis    : fem.Function = None
    n       : fem.Function = None
    Edif    : fem.Function = None
    Edis    : fem.Function = None
    Vdif    : fem.Function = None
    Vdis    : fem.Function = None
    eta     : fem.Function = None
    option_eta : fem.Function = None
    eta_max : float = None
    R        : float = None

def populate_material_properties_rheology(CP,pdb,phase):
    """
    Initialize all rheological properties as FEniCS functions.
    """
    ph = np.int32(phase.x.array)
    P0 = phase.function_space
    
    CP.Bdif    = fem.Function(P0)     ; CP.Bdif.x.array[:]     = pdb.Bdif[ph]
    CP.Bdis    = fem.Function(P0)     ; CP.Bdis.x.array[:]     = pdb.Bdis[ph]
    CP.n       = fem.Function(P0)     ; CP.n.x.array[:]        = pdb.n[ph]
    CP.Edif    = fem.Function(P0)     ; CP.Edif.x.array[:]     = pdb.Edif[ph]
    CP.Edis    = fem.Function(P0)     ; CP.Edis.x.array[:]     = pdb.Edis[ph]
    CP.Vdif    = fem.Function(P0)     ; CP.Vdif.x.array[:]     = pdb.Vdif[ph]
    CP.Vdis    = fem.Function(P0)     ; CP.Vdis.x.array[:]     = pdb.Vdis[ph]
    CP.eta     = fem.Function(P0)     ; CP.eta.x.array[:]      = pdb.eta[ph]
    CP.option_eta = fem.Function(P0)  ; CP.option_eta.x.array[:] = pdb.option_eta[ph]
    CP.eta_max = pdb.eta_max
    CP.R       = pdb.R
    CP.eta.x.scatter_forward()
    return CP 
        
    
# ---------------------------------------------------------------------------------




#----- 
@njit 
def compute_thermal_properties(pdb,T,p,ph):
    
    Cp   = heat_capacity(pdb,T,ph)
    rho  = density(pdb,T,p,ph)
    k    = heat_conductivity(pdb,T,p,rho,Cp,ph)
    
    return Cp, rho, k 


#----------------------------------------------------------------------------------
def heat_conductivity_FX(FG,T,p,Cp,rho):
    """Function that computes the heat conductivity for a given Pressure and Temperature
       k = k0 + kb*exp(-(T-Tr)/kc)+kd*exp(-(T-Tr)/ke)*exp(kf*P) + rad * flag
       if the phase has a constant conductivity, k0 is defined and positive, otherwise is 0.0 
       while while the other properties are set to be 0.0 
       kb*exp(T-Tr/0)=0.0 so no active. 
    Args:
        FG (_type_) : Precomputed function spaces with the material properties
        T (_type_)  : Input temperature
        p (_type_)  : Input pressure
        Cp (_type_) : Cp form 
        rho (_type_): rho form

    Returns:
        k: fem.form
    """
    

    k_rad = FG.A * exp(-(T-FG.T_A)**2/ (2*FG.x_A ** 2 )) + FG.B * exp(-(T - FG.T_B)**2 / (2* FG.x_B**2))

    kappa_lat = FG.k_a + FG.k_b * exp(-(T-FG.Tref)/FG.k_c) + FG.k_d * exp(-(T-FG.Tref)/FG.k_e)
    
    kappa_p   = exp(FG.k_f * p)  

    k = FG.k0  + kappa_lat * kappa_p * Cp * rho + k_rad * FG.fr 

    return k 
#---------------------------------------------------------------------------------
def heat_capacity_FX(FG, T): 

    
    C_p = FG.C0 + FG.C1 * (T**(-0.5)) + FG.C2 * T**(-2.) + FG.C3 * (T**(-3.)) + FG.C4 * T + FG.C5 * T**2

    return C_p
  
def compute_radiogenic(FG, hs): 
    hs.interpolate(FG.radio)
    return hs 
    
#---------------------------------------------------------------------------------
@njit
def heat_conductivity(pdb,T,p,rho,Cp,ph):    


    k_rad = pdb.A * np.exp(-(T-pdb.T_A)**2/ (2*pdb.x_A ** 2 )) + pdb.B * np.exp(-(T - pdb.T_B)**2 / (2* pdb.x_B**2))

    kappa_lat = pdb.k_a[ph] + pdb.k_b[ph] * np.exp(-(T-pdb.Tref)/pdb.k_c[ph]) + pdb.k_d[ph] * np.exp(-(T-pdb.Tref)/pdb.k_e[ph])
    
    kappa_p   = np.exp(pdb.k_f[ph] * p)  

    k = pdb.k0[ph] + kappa_lat * kappa_p * Cp * rho + k_rad * pdb.radio_flag[ph]

    return k 
#---------------------------------------------------------------------------------

@njit
def heat_capacity(pdb,T,ph):

    C_p = pdb.C0[ph] + pdb.C1[ph] * (T**(-0.5)) + pdb.C2[ph] * T**(-2.0) + pdb.C3[ph] * (T**(-3.)) + pdb.C4[ph]* T + pdb.C5[ph] * T**2
        
    return C_p
#---------------------------------------------------------------------------------
@njit
def density(pdb,T,p,ph):
    rho_0 = pdb.rho0[ph] 
    
    if pdb.option_rho[ph] == 0:
        # constant variables 
        return rho_0 
    else :
        # calculate rho
        rho     = rho_0 * (1 - np.exp(- p * pdb.alpha2[ph])*( pdb.alpha0[ph] * (T - pdb.Tref) + (pdb.alpha1[ph]/2.) * ( T**2 - pdb.Tref**2 )))
        if pdb.option_rho[ph] == 2:
            # calculate the pressure dependence of the density
            Kb = pdb.Kb[ph]
            rho = rho * np.exp(p/Kb)    
    
    return rho

#-----
def density_FX(FG, T, p):
    """
    Compute density as a UFL expression, FEniCSx-compatible.
    """


    # Base density (with temperature dependence)
    temp_term = exp(- p * FG.alpha2)*(FG.alpha0 * (T - FG.Tref) + (FG.alpha1 / 2.0) * (T**2 - FG.Tref**2))
    rho_temp = FG.rho0 * (1-temp_term) 

    # Add pressure dependence if needed
    rho = conditional(
        eq(FG.option_rho, 0), FG.rho0,
        conditional(
            eq(FG.option_rho, 1), rho_temp,
            rho_temp * exp(p / FG.Kb)
        )
    )

    return rho 
#----------------------------------


#----------------------------------
def alpha_FX(FG, T, p):
    """
    Compute density as a UFL expression, FEniCSx-compatible.
    """

    # Base density (with temperature dependence)
    alpha =  exp(- p * FG.alpha2) * (FG.alpha0  + (FG.alpha1) * (T- FG.Tref))


    return alpha 


#-----------------------------------
def compute_viscosity_FX(e,T_in,P_in,FR,sc):
    """
    It is wrong, but frequently used: it does not change too much by the end the prediction. 
    I use the minimum viscosity possible. The alternative is taking the average between eta_min and eta_av. So, 
    since I do not understand how I can easily integrate a full local iteration, I prefer to use the "wrong" composite method
    """    
    def ufl_pow(u, v, eps=0):
        return ufl.exp(v * ufl.ln(u + eps))
    
    def compute_eII(e):
        e_II  = sqrt(0.5*inner(e, e) + 1e-15)    
        return e_II
    
    e_II = compute_eII(e)
    # If your phase IDs are available per cell for mesh0:
    
    # Eta max 
    # strain indipendent  
    cdf = FR.Bdif * exp(-(FR.Edif + P_in * sc.stress * FR.Vdif )/(FR.R * T_in * sc.Temp)) ; cds = FR.Bdis * exp(-(FR.Edis + P_in * sc.stress * FR.Vdis)/(FR.R * T_in*sc.Temp))
    # compute tau guess
    n_co  = (1-FR.n)/FR.n
    n_inv = 1/FR.n 
    # Se esiste un cazzo di inferno in culo a Satana ci vanno quelli che hanno generato 
    # sto modo creativo di fare gli esponenti. 
    etads     = 0.5 * cds**(-n_inv) * e_II**n_co
    etadf     = 0.5 * cdf**(-1)
    eta_av    = 1 / (1 / etads + 1/etadf + 1/FR.eta_max)
    eta_df    = 1 / (1 / etadf + 1 / FR.eta_max) 
    eta_ds    = 1 / (1 / etads + 1 / FR.eta_max)
    
    
    
    # check if the option_eta -> constant or not, otherwise release the composite eta 
    eta = ufl.conditional(
        ufl.eq(FR.option_eta, 0.0), FR.eta,
        ufl.conditional(
            ufl.eq(FR.option_eta, 1.0), eta_df,
            eta_av
        )
    )


    return eta

def compute_plastic_strain(e_II,T_in,P_in,pdb,ph,phwz,sc):
    """
    It is wrong, but frequently used: it does not change too much by the end the prediction. 
    I use the minimum viscosity possible. The alternative is taking the average between eta_min and eta_av. So, 
    since I do not understand how I can easily integrate a full local iteration, I prefer to use the "wrong" composite method
    """    

    e_II = e_II + 1e-15
    
    
    # If your phase IDs are available per cell for mesh0:
    
    # UNFORTUNATELY I AM STUPID and i do not have any idea how to scale the energies such that it would be easier to handle. Since the scale of force and legth is self-consistently related to mass, i do not know how to deal with the fucking useless mol in the energy of activation 
    T = T_in.copy()
    P = P_in.copy()
    T.x.array[:] = T.x.array[:]*sc.Temp  ;T.x.scatter_forward()
    P.x.array[:] = P.x.array[:]*sc.stress;P.x.scatter_forward()
    P0    = T.function_space
    
    # Gather material parameters as UFL expressions via indexing
    Bdif    = fem.Function(P0,name = 'Bdif')  ; Bdif.x.array[:]    =  pdb.Bdif[phwz]
    Bdis    = fem.Function(P0,name = 'Bdis')  ; Bdis.x.array[:]    =  pdb.Bdis[phwz]
    n       = fem.Function(P0,name = 'n')     ; n.x.array[:]       =  pdb.n[phwz]
    Edif    = fem.Function(P0,name = 'Edif')  ; Edif.x.array[:]    =  pdb.Edif[phwz]
    Edis    = fem.Function(P0,name = 'Edis')  ; Edis.x.array[:]    =  pdb.Edis[phwz]
    Vdif    = fem.Function(P0,name = 'Vdif')  ; Vdif.x.array[:]    =  pdb.Vdif[phwz]
    Vdis    = fem.Function(P0,name = 'Vdis')  ; Vdis.x.array[:]    =  pdb.Vdis[phwz]
    


    # In case the viscosity for the given phase is constant 
    eta_con     = fem.Function(P0) ; eta_con.x.array[:]     =  pdb.eta[phwz]
    # Option for eta for a given marker number ph 
    opt_eta = fem.Function(P0)  ; opt_eta.x.array[:] =  pdb.option_eta[phwz]
    # Eta max 
    Bd_max  = 1 / 2 / pdb.eta_max
    # strain indipendent  
    cdf = Bdif * exp(-(Edif + P * Vdif )/(pdb.R * T)) ; cds = Bdis * exp(-(Edis + P * Vdis)/(pdb.R * T))
    # compute tau guess
    n_co  = (1-n)/n
    n_inv = 1/n 
    # Se esiste un cazzo di inferno in culo a Satana ci vanno quelli che hanno generato 
    # sto modo creativo di fare gli esponenti. 
    etads     = 0.5 * cds**(-n_inv) * e_II**n_co
    etadf     = 0.5 * cdf**(-1)
    eta_av    = 1 / (1 / etads + 1/etadf + 1/pdb.eta_max)
    
    # -> Compute the tau lim 
    tau_lim  = pdb.cohesion * cos(pdb.friction_angle) + P_in * sin (pdb.friction_angle)
    
    tau_vis  = 2 * eta_av * e_II
    
    
    # check if the option_eta -> constant or not, otherwise release the composite eta 
    
    tau_eff = ufl.conditional(tau_vis > tau_lim, tau_lim, tau_vis)

    e_plr2    = (e_II - (tau_eff / 2 /eta_av)) / e_II
    
    
    e_plr = ufl.conditional(e_plr2 < 0.0, 0.0, e_plr2)


    return e_plr, tau_eff

#-----------------------------------------------------------------------------------
#@njit
def compute_viscosity(e,T,P,B,n,E,V,R):

    t = -1+1/n

    eta = 0.5*B**(-1/n)*e**t*np.exp((E+P*V)/(n*R*T))

    return eta 
#---------------------------------------------------------------------------------
# TILL STOKES BETTER TO NOT TOUCH THIS FUNCTION
#@njit
def viscosity(exx:float,eyy:float,exy:float,T:float,P:float,p_data,it:int,ph:int):#,imat):
    ck = 0
    e=np.sqrt(0.5*(exx**2+eyy**2)+exy**2)
    
    
    
    rheo_type = p_data.option_eta[ph]
    eta_max   = p_data.eta_max
    eta_min   = p_data.eta_min
    Bdif = p_data.Bdif[ph]
    Edif = p_data.Edif[ph]
    Vdif = p_data.Vdif[ph]

    R    = p_data.R
    Bdis = p_data.Bdis[ph]
    Edis = p_data.Edis[ph]
    Vdis = p_data.Vdis[ph]
    n    = p_data.n[ph]
    T = p_data.T_Scal * T 
    P = p_data.P_Scal * P
    # Short explanation: I do not know how to not dimensionalise E, R and V all these quantities have 
    # a dependency with mol which is a measure of mass, but, also specific to the chemical composition 
    # of the element -> so, in principle, we can derive a measure in E/kg R -> E/kg and so forth 
    # However seems arbitrary. Since the exponential exp(E+PV/RT)=> is by default dimensionless, I simply 
    # unscale the temperature and pressure for computing the temperature dependency. In alterative -> I scale m^3/mol and 
    # E/mol with the usual scale leaving E* V* to be 1/mol. This is an alternative strategy, but seems annoying. 

    
    # Certain area of the model does not have a real velocity field
    if e == 0: 
        e = 1e-21
    
    if (it == 0) or (rheo_type== 0):
        return p_data.eta_def
    
    if (rheo_type) == 0: 
        return p_data.eta[ph]

          
    if rheo_type== 1:
        # diffusion creep 
        eta_dif = compute_viscosity(e,T,P,Bdif,1,Edif,Vdif,R)

        val=1/(1/eta_max+1/eta_dif)
        
    if rheo_type == 3: 
        # dislocation creep 
        eta_dis = compute_viscosity(e,T,P,Bdis,n,Edis,Vdis,R)
        val=1/(1/eta_max+1/eta_dis)
        
    if rheo_type == 2:
        # dislocation & diffusion creep 
        eta_dif = compute_viscosity(e,T,P,Bdif,1,Edif,Vdif,R)
        eta_dis = compute_viscosity(e,T,P,Bdis,n,Edis,Vdis,R)
        eta_eff = (1/eta_dif+1/eta_dis)**(-1)
        val=1/(1/eta_max+1/eta_eff)
    
    if rheo_type == 4: 

        val,tau = point_iteration(eta_max,eta_min,e,T,P,Bdif,Edif,Vdif,Bdis,n,Edis,Vdis,R)
          
    return val
#---------------------------------------------------------------------------------
# Strain Partitioning 
def strain_partitioning(e:float,T:float,P:float,p_data,ph:int):#,imat):
    ck = 0
    
    
    rheo_type = p_data.option_eta[ph]
    eta_max   = p_data.eta_max
    eta_min   = p_data.eta_min
    Bdif = p_data.Bdif[ph]
    Edif = p_data.Edif[ph]
    Vdif = p_data.Vdif[ph]

    R    = p_data.R
    Bdis = p_data.Bdis[ph]
    Edis = p_data.Edis[ph]
    Vdis = p_data.Vdis[ph]
    n    = p_data.n[ph]
    T =  T 
    P =  P

    e_pl,e_dis, tau = point_iteration2(eta_max,eta_min,e,T,P,Bdif,Edif,Vdif,Bdis,n,Edis,Vdis,R,p_data.friction_angle)
          
    return e_pl,e_dis, tau



#---------------------------------------------------------------------------------

#@njit
def _find_tau_guess(cdf,cds,n,eta_max,e):
    """
    input:
    n -> stress exponent
    Bdisl -> dislocation creep parameter
    Qdisl -> dislocation activation energy
    Bdif  -> diffusion creep parameter
    Qdif  -> diffusion activation energy
    eta_max -> maximum viscosity
    e -> strain rate
    T -> temperature
    output:
    t -> tau_guess
    => First compute the minimum and maximum viscosity for the given 
    input strain rate and temperature.
    -> Then compute the relative stresses associated with these mecchanism 
    -> compute the average value of the stresses
    ---> The real stress is between the value defined by the armonic average viscosity and the 
    value defined by the minimum viscosity between the involved mechanism 
    """
    # Compute the viscosity associated with the dislocation and diffusion creep
    # mechanism
    etads=0.5*cds**(-1/n)*e**((1-n)/n)
    etadf=0.5*cdf**(-1)
    # Compute the viscosity associated with the maximum viscosity
    # mechanism
    av_vis = (1/etadf+1/etads+1/eta_max)**(-1)
    
    if etadf<etads:
        min_vis = etadf
    else:
        min_vis = etads

    if min_vis>eta_max:
        min_vis = eta_max


    tau_min = 2*av_vis*e
    tau_max = 2*min_vis*e 

    return tau_min,tau_max


#---------------------------------------------------------------------------------

# Define the function f(tau) for root-finding
##@njit
def f(tau, compliance_disl, compliance_diff, B_max, e, n):
    """Equation to solve for tau."""
    return (e - (compliance_disl * tau**n + compliance_diff * tau + B_max * tau)) / e

# Custom bisection method
#@njit
def bisection_method(a, b, tol, max_iter, compliance_disl, compliance_diff, B_max, e, n):
    """Perform bisection method to find the root of f in the interval [a, b]."""
    fa = f(a, compliance_disl, compliance_diff, B_max, e, n)
    fb = f(b, compliance_disl, compliance_diff, B_max, e, n)
    if (np.abs(fa)<tol) or (np.abs(fb)<tol):
        return a if np.abs(fa)<tol else b
    if fa * fb > 0:
        raise ValueError("Function has the same sign at the endpoints a and b")

    for _ in range(max_iter):
        # Compute midpoint
        c = (a + b) / 2
        fc = f(c, compliance_disl, compliance_diff, B_max, e, n)

        # Check for convergence
        if abs(fc) < tol:
            return c
        elif fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return (a + b) / 2  # Return midpoint after max_iter iterations
#---------------------------------------------------------------------------------

# Main point_iteration function
#@njit
def point_iteration(eta_max: float,
                    eta_min: float,
                    e: float,
                    T: float,
                    P: float,
                    Bdif: float,
                    Edif: float,
                    Vdif: float,
                    Bdis: float,
                    n: float,
                    Edis: float,
                    Vdis: float,
                    R: float) -> float:
    """
    Optimized point-wise iteration to compute viscosity, real stress.
    """

    # Precompute constant values to avoid repeated calculations
    B_max = 1 / (2 * eta_max)
    compliance_disl = Bdis * np.exp(-(Edis+P*Vdis) / (R * T))
    compliance_diff = Bdif * np.exp(-(Edif+P*Vdif) / (R * T))

    # Find tau_guess using custom bisection method
    tau_min, tau_max = _find_tau_guess(compliance_diff, compliance_disl, n, eta_max, e)
    if abs(tau_max - tau_min) / (tau_max + tau_min) < 1e-6:
        tau = (tau_min + tau_max) / 2
    else:
        tol = 1e-8
        max_iter = 100
        tau = bisection_method(tau_min, tau_max, tol, max_iter, compliance_disl, compliance_diff, B_max, e, n)

    # Compute the viscosity and real stress
    e_diff = compliance_diff * tau
    e_dis = compliance_disl * tau ** n
    e_max = B_max * tau

    tau_total = tau + 2 * eta_min * e
    eta = tau_total / (2 * e)

    return eta, tau_total
#---------------------------------------------------------------------------------

# Main point_iteration function
#@njit
def point_iteration2(eta_max: float,
                    eta_min: float,
                    e: float,
                    T: float,
                    P: float,
                    Bdif: float,
                    Edif: float,
                    Vdif: float,
                    Bdis: float,
                    n: float,
                    Edis: float,
                    Vdis: float,
                    R: float,
                    phi: float) -> float:
    """
    Optimized point-wise iteration to compute viscosity, real stress.
    """

    # Precompute constant values to avoid repeated calculations
    B_max = 1 / (2 * eta_max)
    compliance_disl = Bdis * np.exp(-(Edis+P*Vdis) / (R * T))
    compliance_diff = Bdif * np.exp(-(Edif+P*Vdif) / (R * T))
    tau_lim = 10e6*np.cos(5*pi/180)+np.sin(phi)*P
    # Find tau_guess using custom bisection method
    tau_min, tau_max = _find_tau_guess(compliance_diff, compliance_disl, n, eta_max, e)
    if tau_min >= tau_lim:
        tau = tau_lim
    elif abs(tau_max - tau_min) / (tau_max + tau_min) < 1e-6:
        tau = (tau_min + tau_max) / 2
    else:
        tol = 1e-12
        max_iter = 100
        tau = bisection_method(tau_min, tau_max, tol, max_iter, compliance_disl, compliance_diff, B_max, e, n)


    # Compute the viscosity and real stress
    e_diff = compliance_diff * tau
    e_dis = compliance_disl * tau ** n
    e_max = B_max * tau
    e_pl  = e-(e_diff+e_dis+e_max) 
    e_vs = e_diff+e_dis+e_max 
    

    return e_pl, e_vs, tau












def unit_test_thermal_properties(pt_save):    
    """
    Unit test for thermal properties functions. Function that the author used to test the thermal properties functions and debug -> by the end
    I will introduce a few folder in which the data will be saved for being benchmarked in other system and being sure that the code is worked as
    expected. On the other hand, there are a few shit with fenicx that I need to account for, fuck. 
    """
    import numpy as np 
    import sys, os
    sys.path.append(os.path.abspath("src"))
    from phase_db import PhaseDataBase 
    from phase_db import _generate_phase
    from scal import Scal
    
    T = np.linspace(298.15,273.15+1500,num=1000) # Vector temperature
    P = np.linspace(0.0, 15, num = 1000) * 1e9  # Vector pressure   
    
    # phase data base for the test 
    
    pdb = PhaseDataBase(1)
    pdb = _generate_phase(pdb,0,option_rho = 2,option_rheology = 0, option_k = 3, option_Cp = 3,eta=1e21)
    
    rho_test = np.zeros([len(T),len(P)],dtype = np.float64); k_test = np.zeros_like(rho_test); Cp_test = np.zeros_like(rho_test)
    
    for i in range(len(T)):
        for j in range(len(P)):
            rho_test[i,j] = density(pdb,T[i],P[j],0)
            k_test[i,j]   = heat_conductivity(pdb,T[i],P[j],0)
            Cp_test[i,j]  = heat_capacity(pdb,T[i],P[j],0)
    
    thermal_diffusivity = k_test/rho_test/Cp_test
    
    base_cmap = plt.get_cmap('inferno')
    from matplotlib.colors import ListedColormap

    # Number of discrete colors
    N = 20

    # Create a new discrete colormap
    colors = base_cmap(np.linspace(0, 1, N))
    discrete_cmap = ListedColormap(colors)
    
    
    fig = plt.figure()
    ax = fig.gca()
    a = ax.pcolormesh(T-273.15,P,rho_test.T,shading='gouraud',cmap = discrete_cmap)
    plt.colorbar(a,label=r'$\varrho$ [$\frac{\mathrm{kg}}{\mathrm{m}^3}$]')
    ax.set_xlabel(r'T [${\circ}^{C}$]')    
    ax.set_ylabel(r'P [GPa]')
    fig.savefig("%s/density.png"%pt_save)      
    
    
        
    fig = plt.figure()
    ax = fig.gca()
    a = ax.pcolormesh(T-273.15,P/1e9,rho_test.T,shading='gouraud',cmap = discrete_cmap)
    plt.colorbar(a,label=r'$\varrho$ [$\frac{\mathrm{kg}}{\mathrm{m}^3}$]')
    ax.set_xlabel(r'T [^${\circ}{C}$]')    
    ax.set_ylabel(r'P [GPa]')
    fig.savefig("%s/density.png"%pt_save)     
    
    
        
    fig = plt.figure()
    ax = fig.gca()
    a = ax.pcolormesh(T-273.15,P/1e9,k_test.T,shading='gouraud',cmap = discrete_cmap)
    plt.colorbar(a,label=r'$k$ [$\frac{\mathrm{W}}{\mathrm{m}\mathrm{K}}$]')
    ax.set_xlabel(r'T [^${\circ}{C}$]')    
    ax.set_ylabel(r'P [GPa]')
    fig.savefig("%s/conductivity.png"%pt_save)     
    
    fig = plt.figure()
    ax = fig.gca()
    a = ax.pcolormesh(T-273.15,P/1e9,Cp_test.T,shading='gouraud',cmap = discrete_cmap)
    plt.colorbar(a,label=r'$C_p$ [$\frac{\mathrm{J}}{\mathrm{kg}\mathrm{K}}$]')
    ax.set_xlabel(r'T [^${\circ}{C}$]')    
    ax.set_ylabel(r'P [GPa]')
    fig.savefig("%s/capacity.png"%pt_save)  
    
    fig = plt.figure()
    ax = fig.gca()
    a = ax.pcolormesh(T-273.15,P/1e9,np.log10(thermal_diffusivity.T),shading='gouraud',cmap = discrete_cmap)
    plt.colorbar(a,label=r'$\kappa$ [$\frac{\mathrm{m^2}}{\mathrm{s}}$]')
    ax.set_xlabel(r'T [^${\circ}{C}$]')    
    ax.set_ylabel(r'P [GPa]')
    fig.savefig("%s/diffusivity.png"%pt_save)  
        
    
    plt.close('all')
    return rho_test, k_test, Cp_test 
    
    
def unit_test_thermal_properties_scaling(pt_save,dim_rho,dim_k,dim_Cp):    
    """
    Unit test for thermal properties functions. Function that the author used to test the thermal properties functions and debug -> by the end
    I will introduce a few folder in which the data will be saved for being benchmarked in other system and being sure that the code is worked as
    expected. On the other hand, there are a few shit with fenicx that I need to account for, fuck. 
    """
    import numpy as np 
    import sys, os
    sys.path.append(os.path.abspath("src"))
    from phase_db import PhaseDataBase 
    from phase_db import _generate_phase
    from scal import Scal
    from scal import _scaling_material_properties

    sc = Scal(L=660e3, eta = 1e21, Temp = 1350, stress = 2e9) 
    
    T = np.linspace(298.15,273.15+1500,num=1000) # Vector temperature
    P = np.linspace(0.0, 15, num = 1000) * 1e9  # Vector pressure   
    
    T /= sc.Temp 
    P /= sc.stress 

    
    
    # phase data base for the test 
    
    pdb = PhaseDataBase(1)
    pdb = _generate_phase(pdb,0,option_rho = 2,option_rheology = 0, option_k = 3, option_Cp = 3,eta=1e21)
    
    
    pdb = _scaling_material_properties(pdb,sc)
    
    rho_test = np.zeros([len(T),len(P)],dtype = np.float64); k_test = np.zeros_like(rho_test); Cp_test = np.zeros_like(rho_test)
    
    for i in range(len(T)):
        for j in range(len(P)):
            rho_test[i,j] = density(pdb,T[i],P[j],0)
            k_test[i,j]   = heat_conductivity(pdb,T[i],P[j],0)
            Cp_test[i,j]  = heat_capacity(pdb,T[i],P[j],0)
    
    thermal_diffusivity = k_test/rho_test/Cp_test
    
    base_cmap = plt.get_cmap('inferno')
    from matplotlib.colors import ListedColormap

    # Number of discrete colors
    N = 20

    # Create a new discrete colormap
    colors = base_cmap(np.linspace(0, 1, N))
    discrete_cmap = ListedColormap(colors)
    
    
        
    fig = plt.figure()
    ax = fig.gca()
    a = ax.pcolormesh(T,P,rho_test.T,shading='gouraud',cmap = discrete_cmap)
    plt.colorbar(a,label=r'$\varrho^{\dagger}$')
    ax.set_xlabel(r'$T^{\dagger}$')    
    ax.set_ylabel(r'$P^{\dagger}$')
    fig.savefig("%s/density_ND.png"%pt_save)     
    
    
        
    fig = plt.figure()
    ax = fig.gca()
    a = ax.pcolormesh(T,P,k_test.T,shading='gouraud',cmap = discrete_cmap)
    plt.colorbar(a,label=r'$k^{\dagger}$')
    ax.set_xlabel(r'$T^{\dagger}$')    
    ax.set_ylabel(r'$P^{\dagger}$')
    fig.savefig("%s/conductivity_ND.png"%pt_save)     
    
    fig = plt.figure()
    ax = fig.gca()
    a = ax.pcolormesh(T,P,Cp_test.T,shading='gouraud',cmap = discrete_cmap)
    plt.colorbar(a,label=r'$C_p^{\dagger}$')
    ax.set_xlabel(r'$T^{\dagger}$')    
    ax.set_ylabel(r'$P^{\dagger}$')
    fig.savefig("%s/capacity_ND.png"%pt_save)  
    
    fig = plt.figure()
    ax = fig.gca()
    a = ax.pcolormesh(T,P,np.log10(thermal_diffusivity.T),shading='gouraud',cmap = discrete_cmap)
    plt.colorbar(a,label=r'$\kappa^{\dagger}$')
    ax.set_xlabel(r'$T^{\dagger}$')    
    ax.set_ylabel(r'$P^{\dagger}$')
    fig.savefig("%s/diffusivity_ND.png"%pt_save)  

    plt.close('all')
    k_test   *= sc.k 
    rho_test *= sc.rho 
    Cp_test  *= sc.Cp 
    res_Cp   = np.linalg.norm(Cp_test-dim_Cp,2)/np.linalg.norm(Cp_test+dim_Cp,2)
    res_k    = np.linalg.norm(k_test-dim_k,2)/np.linalg.norm(k_test+dim_k,2)
    res_rho  = np.linalg.norm(rho_test-dim_rho,2)/np.linalg.norm(rho_test+dim_rho,2)
    
    print('    { => Test <= }')
    print('      res Cp   %.4e'%res_Cp)
    print('      res rho  %.4e'%res_rho)
    print('      res k    %.4e'%res_k)
    print('    { <= Test => }') 
    tol = 1e-12
    if res_Cp > tol or res_k > tol or res_rho > tol : 
        raise('Something wrong, wrong scaling, wrong computer, wrong everything')
    else: 
        assert('Pass')
    
    return 0    

    
    
    
def unit_test_viscosity(pt_save): 
    
    import numpy as np 
    import sys, os
    sys.path.append(os.path.abspath("src"))
    from phase_db import PhaseDataBase 
    from phase_db import _generate_phase
    from scal import Scal
    from scal import _scaling_material_properties

    #sc = Scal(L=660e3, eta = 1e21, Temp = 1350, stress = 2e9) 
    
    
    T = np.linspace(298.15,273.15+1500,num=1000) # Vector temperature
    
    P = np.linspace(0.0, 15, num = 1000) * 1e9  # Vector pressure   
    # Strain rate {required}
    
    exx, eyy, exy = 1e-14,1e-14,1e-14 
    
    pdb = PhaseDataBase(1)
    pdb = _generate_phase(pdb,0,option_rho = 2,option_rheology = 4, option_k = 3, option_Cp = 3,eta=1e21,name_diffusion='Van_Keken_diff',name_dislocation='Van_Keken_disl')
    
    eta_dim = np.zeros([len(T),len(P)],dtype = np.float64)
    eta_nd  = np.zeros([len(T),len(P)],dtype = np.float64)

    for i in range(len(T)):
        for j in range(len(P)):#exx:float,eyy:float,exy:float,T:float,P:float,p_data,it:int,ph:int
            eta_dim[i,j] = viscosity(exx,eyy,exy,T[i],P[i],pdb,1,0)
    
    
    
    from matplotlib.colors import ListedColormap
    import cmcrameri as cmc
    base_cmap = plt.get_cmap('cmc.oslo')

    # Number of discrete colors
    N = 20

    # Create a new discrete colormap
    colors = base_cmap(np.linspace(0, 1, N))
    discrete_cmap = ListedColormap(colors)
    
    
    
    fig = plt.figure()
    ax = fig.gca()
    a = ax.pcolormesh(T-273.15,P/1e9,np.log10(eta_dim.T),shading='gouraud',cmap = discrete_cmap)
    plt.colorbar(a,label=r'$\eta_{eff}$ [Pas]')
    ax.set_xlabel(r'T [^${\circ}{C}$]')    
    ax.set_ylabel(r'P [GPa]')
    fig.savefig("%s/viscosity.png"%pt_save)
        
    sc = Scal(L=660e3, eta = 1e21, Temp = 1350, stress = 2e9) 
    
    T /= sc.Temp 
    P /= sc.stress 
    exx/= sc.strain ; exy /=sc.strain; eyy /= sc.strain  
    pdb = _scaling_material_properties(pdb,sc)
     
    for i in range(len(T)):
        for j in range(len(P)):#exx:float,eyy:float,exy:float,T:float,P:float,p_data,it:int,ph:int
            eta_nd[i,j] = viscosity(exx,eyy,exy,T[i],P[i],pdb,1,0)
    
    err = np.linalg.norm(eta_dim - eta_nd*sc.eta,2)/np.linalg.norm(eta_dim + eta_nd*sc.eta,2)
    print('Error is %.4e' %err)
    if err > 1e-12:
        raise ('Something wrong')
    else: 
        fig = plt.figure()
        ax = fig.gca()
        a = ax.pcolormesh(T,P,np.log10(eta_nd.T),shading='gouraud',cmap = discrete_cmap)
        plt.colorbar(a,label=r'$\eta_{eff}^{\dagger}$')
        ax.set_xlabel(r'$T^{\dagger}$')    
        ax.set_ylabel(r'$P^{\dagger}$')
        fig.savefig("%s/viscosity.png"%pt_save)
    
    
    
    
    




"""
from pathlib import Path

folder_path = Path("your_folder_name")
folder_path.mkdir(parents=True, exist_ok=True)
"""

def _frictional_alternative(pt_save):
    import numpy as np 
    import sys, os
    sys.path.append(os.path.abspath("src"))
    from phase_db import PhaseDataBase 
    from phase_db import _generate_phase
    from scal import Scal
    from scal import _scaling_material_properties
    from scipy import special 

    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",       # serif = Computer Modern
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 14,
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    })
    
    
    z           = np.linspace(0,300e3,300)
    rho         = 3000.0
    Pl          = 9.81*rho*z     
    Cp          = 1250
    k           = 3.0
    kappa       = k/rho/Cp 
    t           = 50e6*365.25*24*60*60
    T           = 20+(1350-20) * special.erf(z /2 /np.sqrt(t * kappa))+273.15
    
    lit   = 50e3
    dec   = 100e3 
    creep = 40e3 
       
    sub_channel = 1e3 
    sub_channel2 = np.linspace(500,10e3,1000)
    
    jtanh       = np.zeros(len(z))
    m           = (lit+dec)/2
    ls          = (dec-m)
    jtanh       = 1-0.5 * ((1)+(1)*np.tanh((z-dec)/(ls/4)))
    
    v_slab = 1.0/100/365.25/24/60/60


    # Model Van Keken: From what I do understand from these war criminals 
    
    friction_coef = 0.06
        
    frictional_heatHob = jtanh * v_slab * Pl * friction_coef # Hobson uses only the jump function
    
           
    pdb = PhaseDataBase(1,atan(friction_coef))
    pdb = _generate_phase(pdb,0,option_rho = 2,option_rheology = 4, option_k = 3, option_Cp = 3,eta=1e21,name_diffusion='Hirth_Wet_Olivine_diff',name_dislocation='Hirth_Wet_Olivine_disl')

    e_pl = np.zeros([len(z),len(sub_channel2)]); e_vs = np.zeros([len(z),len(sub_channel2)]);tau = np.zeros([len(z),len(sub_channel2)])
    eplr = np.zeros([len(z),len(sub_channel2)]); evsr = np.zeros([len(z),len(sub_channel2)]);frictional_heating2=np.zeros([len(z),len(sub_channel2)])
    max_depth_plastic = np.zeros(len(sub_channel2))
    for k in range(len(sub_channel2)):
        strain_rate = (v_slab * jtanh)/sub_channel2[k] # Compute the strain rate 
        
        for i in range(len(z)):
           e_pl[i,k],e_vs[i,k],tau[i,k] = strain_partitioning(strain_rate[i],T[i],Pl[i],pdb,0)
           if np.log10(e_pl[i,k]) < -20: 
                e_pl[i,k] = 0.0
           elif np.log10(e_vs[i,k]) < -20:  
               e_vs[i,k] = 0.0
        

        eplr[:,k] = e_pl[:,k]/strain_rate ; evsr[:,k] = e_vs[:,k]/strain_rate 
        frictional_heating2[:,k]  = eplr[:,k] * Pl * v_slab * jtanh * friction_coef #+ evsr[:,k] * tau[:,k] * strain_rate * sub_channel2[k] 
        a = np.where(frictional_heating2[:,k]==np.nanmax(frictional_heating2[:,k]))[0][0]
        max_depth_plastic[k] = z[a]



    

    

    fig = plt.figure()
    ax0 = plt.gca()
    ax0.plot(frictional_heatHob,-z/1e3,c='forestgreen',linestyle=':',linewidth=1.0, label = 'Van Keken 2018')
    #ax0.plot(strain_rate,-z/1e3,c='forestgreen',linewidth=1.0)
    ax0.set_xlabel(r'Frictional heat flux $\frac{W}{m^2}$')
    ax0.set_ylabel(r'Depth $[km]$')
    ax0.legend()
    ax0.set_xscale('linear')
    ax0.set_ylim([-100,0.0])


    
    
    fig = plt.figure()
    ax0 = plt.gca()
    ax0.plot(e_pl,-z/1e3,c='k',linestyle=':',linewidth=0.8, label=r'$\dot{\varepsilon}_{pl}$')
    ax0.plot(e_vs,-z/1e3,c='r',linestyle='-.',linewidth=0.8,label=r'$\dot{\varepsilon}_{vs}$')
    ax1 = ax0.twiny()
    ax1.plot(tau/1e6,-z/1e3,c='b',linewidth=1.0,label=r'$\tau$ [MPa]')
    #ax0.plot(strain_rate,-z/1e3,c='forestgreen',linewidth=1.0)
    ax1.set_xlabel(r'${\tau}$ [MPa]')
    ax0.set_xlabel(r'$\dot{\varepsilon}_{II}$ [$\frac{1}{s}$]')
    ax0.set_ylabel(r'Depth $[km]$')
    ax0.legend()
    ax1.legend()
    ax0.set_xscale('log')
    ax0.set_ylim([-100,0.0])

    
    fig = plt.figure()
    ax0 = plt.gca()
    ax0.plot(frictional_heating2,-z/1e3,c='r',linestyle='-.',linewidth=0.8,label=r'$f_2(T,P,\varepsilon)$')
    #ax1 = ax0.twiny()
    #ax1.plot(tau/1e6,-z/1e3,c='b',linewidth=1.0,label=r'$\tau$ [MPa]')
    #ax0.plot(strain_rate,-z/1e3,c='forestgreen',linewidth=1.0)
    #ax1.set_xlabel(r'${\tau}$ [MPa]')
    ax0.set_xlabel(r'Frictional heat flux $\frac{W}{m^2}$')
    ax0.set_ylabel(r'Depth $[km]$')
    ax0.legend()
    #ax1.legend()
    ax0.set_ylim([-100,0.0])

    fig = plt.figure()
    ax0 = plt.gca()
    ax0.plot(frictional_heating2,-z/1e3,c='b',linestyle='-.',linewidth=1.0,label=r'$f_2(T,P,\varepsilon)$')
    ax0.plot(frictional_heatHob,-z/1e3,c='forestgreen',linestyle=':',linewidth=1.0, label = 'Hobson and May 2025')
    #ax1 = ax0.twiny()
    #ax1.plot(tau/1e6,-z/1e3,c='b',linewidth=1.0,label=r'$\tau$ [MPa]')
    #ax0.plot(strain_rate,-z/1e3,c='forestgreen',linewidth=1.0)
    #ax1.set_xlabel(r'${\tau}$ [MPa]')
    ax0.set_xlabel(r'Frictional heat flux $\frac{W}{m^2}$')
    ax0.set_ylabel(r'Depth $[km]$')
    ax0.legend()
    #ax1.legend()
    ax0.set_ylim([-100,0.0])    
    
        

    print('bla')
    




if __name__ == '__main__':
    
    import numpy as np 
    import sys, os
    sys.path.append(os.path.abspath("src"))
    from phase_db import PhaseDataBase 
    from phase_db import _generate_phase
    from scal import Scal
    from scal import _scaling_material_properties
    from scipy import special 
    
    pt_save = '../../debug'
    
    if not os.path.exists(pt_save): 
        os.makedirs(pt_save)   

    _frictional_alternative(pt_save)
    
    #rho_dim,k_dim,Cp_dim = unit_test_thermal_properties(pt_save)
    
    #unit_test_thermal_properties_scaling(pt_save,rho_dim,k_dim,Cp_dim)
    


