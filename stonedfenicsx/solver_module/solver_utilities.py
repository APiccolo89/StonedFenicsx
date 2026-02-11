from stonedfenicsx.utils import *
from stonedfenicsx.package_import import *
from stonedfenicsx.scal import Scal

#-------------------------------------------------------------------------       
def decoupling_function(z,fun,g_input):
    """
    Function explanation: 
    [a]: z depth coordinate 
    [b]: fun a function 
    [c]: ctrl => deprecated 
    [d]: D => deprecated 
    [e]: g_input -> still here, geometric parameters
    
    
    
    """
    
    dc = g_input.decoupling
    lit = g_input.ns_depth
    dc = dc/g_input.decoupling
    lit = lit/g_input.decoupling
    z2 = np.abs(z)/g_input.decoupling
    trans = g_input.trans/g_input.decoupling
    

    fun.x.array[:] = 1-0.5 * ((1)+(1)*np.tanh((z2-dc)/(trans/4)))
    
    
    return fun

#---------------------------------------------------------------------------------------------------          
def L2_norm_calculation(f):
    return fem.assemble_scalar(fem.form(ufl.inner(f, f) * ufl.dx))**0.5       
        
#---------------------------------------------------------------------------------------------------       

def compute_adiabatic_initial_adiabatic_contribution(M,T,Tgue,p,FG,vankeken): 
    
    from .compute_material_property import alpha_FX 
    from .utils import evaluate_material_property
    
    
    FS = T.function_space 
    Tg = fem.Function(FS)
    Tg = T.copy()
    v  = ufl.TestFunction(FS)


    if vankeken==0:
        
        res = 1
        while res > 1e-6:
        
            expr = (alpha_FX(FG,Tg,p) * p)/(heat_capacity_FX(FG,Tg) * density_FX(FG,Tg,p))
            a = T * ufl.exp(expr)
            TG1 = evaluate_material_property(a,FS)
            res = compute_residuum(TG1,Tg)
            Tg.x.array[:]  = 0.8*(TG1.x.array[:])+(1-0.8)*Tg.x.array[:]
            

            
        
    else: 
        from .utils import evaluate_material_property
        expr = (alpha_FX(FG,Tg,p) * p)/(heat_capacity_FX(FG,Tg) * density_FX(FG,Tg,p))
        F = T * ufl.exp(expr)
        Tg = evaluate_material_property(F,FS)

    
    return Tg
    
    

def initial_adiabatic_lithostatic_thermal_gradient(sol,lps,FGpdb,M,g,it_outer,ctrl):
    res = 1 
    it = 0
    T_0 = sol.T_O.copy()
    T_Oa = sol.T_O.copy()
    while res > 1e-3: 
        P_old = sol.PL.copy()
        sol = lps.Solve_the_Problem(sol,ctrl,FGpdb,M,g,it_outer,ts=0)
        T_O = compute_adiabatic_initial_adiabatic_contribution(M.domainG,T_0,T_Oa,sol.PL,FGpdb,ctrl.van_keken)
        resp = compute_residuum(sol.PL,P_old)
        resT = compute_residuum(T_O, T_Oa)
        res = np.sqrt(resp**2 + resT**2)
        if it !=0: 
            sol.PL.x.array[:] = 0.8 * sol.PL.x.array[:] + (1-0.8) * P_old.x.array[:]
            sol.PL.x.scatter_forward()
            T_Oa.x.array[:]= T_O.x.array[:] * 0.8 + (1-0.8) * T_O.x.array[:]
            sol.T_O.x.scatter_forward()
        it = it + 1 
        sol.T_O = T_Oa.copy()
        print_ph('Adiabatic res is %.3e'%res)
    
    sol.T_i = T_O.copy()
    sol.T_N = T_O.copy()
    sol.T_O = T_O.copy()
    return sol 

#---------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
def compute_residuum_outer(sol,T,PL,u,p,it_outer,sc,tA,Tmax,ts):
    # Prepare the variables 

    
    
    
    res_u = compute_residuum(sol.u_global,u)
    res_p = compute_residuum(sol.p_global,p)
    res_T = compute_residuum(sol.T_N,T)
    res_PL= compute_residuum(sol.PL,PL)
    
    minMaxU = min_max_array(sol.u_global, vel=True)
    minMaxP = min_max_array(sol.p_global)
    minMaxT = min_max_array(sol.T_N)
    minMaxPL= min_max_array(sol.PL)
    
    # scal back 
    
    minMaxU = minMaxU*(sc.L/sc.T)/sc.scale_vel 
    minMaxP = minMaxP*sc.stress/1e9 
    minMaxT = minMaxT*sc.Temp -273.15
    minMaxPL= minMaxPL*sc.stress/1e9
    
    if minMaxT[1]-(Tmax * sc.Temp-273.15)>1.0: 
        print_ph('Temperature higher than the maximum temperature')
    
    res_total = np.sqrt(res_u**2+res_p**2+res_T**2)
    if not np.isfinite(res_total):
        raise ValueError("res_total is NaN/Inf; check inputs and residual computations.")
    
    time_B_outer = timing.time()

    print_ph('')
    print_ph(f' Outer iteration {it_outer:d} with tolerance {res_total:.3e}, in {time_B_outer-tA:.1f} sec // -- // --->')
    print_ph(f'    []Res velocity       =  {res_u:.3e} [n.d.], max= {minMaxU[1]:.6f}, min= {minMaxU[0]:.6f} [cm/yr]')
    print_ph(f'    []Res Temperature    =  {res_T:.3e} [n.d.], max= {minMaxT[1]:.6f}, min= {minMaxT[0]:.6f} [C]')
    print_ph(f'    []Res pressure       =  {res_p:.3e} [n.d.], max= {minMaxP[1]:.3e}, min= {minMaxP[0]:.3e} [GPa]')
    print_ph(f'    []Res lithostatic    =  {res_PL:.3e}[n.d.], max= {minMaxPL[1]:.3e}, min= {minMaxPL[0]:.3e} [GPa]')
    print_ph(f'    []Res total (sqrt(rp^2+rT^2+rPL^2+rv^2)) =  {res_total:.3e} [n.d.] ')
    print_ph('. =============================================// -- // --->')
    print_ph('')

    sol.mT = np.append(sol.mT,minMaxT[0])
    sol.MT = np.append(sol.MT,minMaxT[1])
    sol.outer_iteration = np.append(sol.outer_iteration,res_total)
    sol.ts = np.append(sol.ts,ts)
    
    
    return res_total, sol 

#------------------------------------------------------------------------------------------------------------

def compute_residuum(a,b):
    dx = a.copy()
    
    dx1 = a.copy()
    
    dx.x.array[:]  = b.x.array[:] - a.x.array[:];dx.x.scatter_forward()
    
    dx1.x.array[:]  = b.x.array[:] + a.x.array[:];dx1.x.scatter_forward()
    
    
    res = L2_norm_calculation(dx)/L2_norm_calculation(dx1)
    
    return res
#------------------------------------------------------------------------------------------------------------
def min_max_array(a, vel = False):
    
    if vel == True: 
        a1 = a.sub(0).collapse()
        a2 = a.sub(1).collapse()
        array = np.sqrt(a1.x.array[:]**2 + a2.x.array[:]**2)
    else: 
        array = a.x.array[:]
    
    local_min = np.min(array[:])
    local_max = np.max(array[:])
    
    global_min = a.function_space.mesh.comm.allreduce(local_min, op=MPI.MIN)
    global_max = a.function_space.mesh.comm.allreduce(local_max, op=MPI.MAX)
    
    return np.array([global_min, global_max],dtype=np.float64)

