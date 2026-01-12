def logistic_function_decoupling():
    import numpy as np
    import matplotlib.pyplot as plt
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
    
    
    
    z = np.linspace(0,300e3,500)
    lit = 50e3
    dec = 100e3 
    creep = 40e3 
    
    # jump function 
    # Linear 
    jfl         = np.zeros(len(z))
    jfl[z<lit]  = 1.0 
    jfl[z>=lit] = 1+(z[z>=lit]-lit)/(lit-dec)
    jfl[z>dec]  = 0.0
    
    # Tanh 
    jtanh      = np.zeros(len(z))
    m          =  dec
    ls         = (10e3)
    jtanh      = 1-0.5 * ((1)+(1)*np.tanh((z-m)/(ls/4)))
    
    fig = plt.figure()
    ax0 = plt.gca()
    ax0.plot(jfl,-z/1e3,c='k',label='linear_function_jump')
    ax0.set_ylabel('Depth [km]')    
    ax0.set_xlabel('friction efficiency [n.d.]')
    ax0.plot(jtanh,-z/1e3,c='firebrick',label='tan_function_jump')
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax0.legend()
    plt.savefig("../../jump_functions.png", dpi=300, bbox_inches='tight')
    plt.close()
    # Friction function 
    # linear 
    frl           = np.zeros(len(z))
    frl[z<creep]  = 1.0 
    frl[z>=creep] = 1+(z[z>=creep]-creep)/(creep-dec)
    frl[z>dec]    = 0 
    
    # Tanh 
    frtan           = np.zeros(len(z))
    m          = (creep+dec)/2
    ls         = 2*(dec-m)
    frtan      = 1-0.5 * ((1)+(1)*np.tanh((z-m)/(ls/4)))
    
    fig = plt.figure()
    ax0 = plt.gca()
    ax0.plot(frl,-z/1e3,c='k',label='linear friction efficiency')
    ax0.set_ylabel('Depth [km]')    
    ax0.set_xlabel('friction efficiency [n.d.]')
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax0.legend()
    plt.savefig("../../efficiency_functions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # case 1 
    eff1 = jfl*frl 
    # case 2 
    eff2 = jfl*frtan

    
    fig, (ax0) = plt.subplots(1, figsize=(10, 6), sharey=True)      
    ax0.plot(eff1,-z/1e3,c='k',label='linearxlinear',linestyle='-.')
    ax0.plot(eff2,-z/1e3,c='r',label='linearxnonlinear',linestyle='-.')
    ax0.set_ylim([-100,0])
    ax0.grid(True, linestyle='--', alpha=0.5)

    ax0.set_ylabel(r'Depth [km]')
    ax0.set_xlabel(r'effective_efficiency [n.d.]')
    ax0.legend()
    plt.savefig("../../effective_function.png", dpi=300, bbox_inches='tight')
    plt.close()
    # W/m3 frictional heating
    pl = 3300.0*9.81*z/1e3
    v  = 5.0/1e2/(365.25*24*60*60)
    c1 = eff1*pl*v 
    c2 = eff2*pl*v
    c3 = eff3*pl*v
    c4 = eff4*pl*v
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)      
    ax0.plot(c1,-z/1e3,c='k',label='linearxlinear',linestyle='-.')
    ax0.plot(c2,-z/1e3,c='r',label='linearxnonlinear',linestyle='-.')
    ax1.plot(c3,-z/1e3,c='forestgreen',label='nonlinearxlinear')
    ax1.plot(c4,-z/1e3,c='firebrick',label='nonlinearxnonlinear')
    ax1.set_ylim([-100,0])
    ax0.set_ylim([-100,0])
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax0.set_ylabel(r'Depth [km]')
    ax1.set_xlabel(r'$\psi$ [$\frac{W}{m^3}$]')
    ax0.set_xlabel(r'$\psi$ [$\frac{W}{m^3}$]')
    ax0.legend()
    ax1.legend()
    plt.savefig("../../Power.png", dpi=300, bbox_inches='tight')
    plt.close()

    dc_u = 100e3
    lit = 50e3
    lit = lit/dc_u
    z = z/dc_u
    dc = 1.0
    zm = (dc + lit)/2
    k = 15.0
    C = 1.0 / (1 + np.exp(-k *(z-zm)))
    C2 = np.zeros(len(z))
    C2[z < lit] = 0.0
    C2[z >= lit] = 1.0/(dc-lit)*(z[z >= lit]-lit)
    C2[C2 > 1.0] = 1.0
    plt.plot(z-lit,C)
    plt.plot(z-lit,C2)
    plt.xlabel('Depth [m]')
    plt.ylabel('Decoupling factor')
    plt.title('Logistic function for decoupling')
    plt.grid()
    plt.show()
    plt.close()
    
    dv = ((1.0 - C)*1e-9)*((z*dc)*9.81*3300)*0.06
    dv2 = ((1.0 - C2)*1e-9)*((z*dc)*9.81*3300)*0.06

    plt.plot((z)*dc_u,dv)
    plt.plot((z)*dc_u,dv2)

    plt.xlabel('Depth [m]')
    plt.ylabel('Decoupling viscous force [Pa]')
    plt.title('Decoupling viscous force')
    plt.grid()
    
    #hobson 
    u = 5.0 
    l = 0.1 
    el = np.linspace(0,300,300)
    c = 100 
    L = 10 
    vel = 0.5 * ((1+0.01)+(1-0.01)*np.tanh((el-c)/(L/4)))
    
    zt = 40e3 
    z0 = 100e3 
    zm = (zt+z0)/2
    zd = (z0-zm)
    z = np.linspace(0,100e3,1000)
    fu = np.zeros(len(z))    
    fu = 1- 0.5 * ((1+0)+(1)*np.tanh((z-zm)/(zd/2)))
    fu2 = np.zeros(len(z))
    fu2[z<zt]  = 1 
    fu2[z>=zt] = 1-(zt-z[z>=zt])/(zt-z0)
    vel = 0.5 * ((1+0.01)+(1-0.01)*np.tanh((z-z0)/(10e3/4)))
    return 0



if __name__ == '__main__':
    logistic_function_decoupling()