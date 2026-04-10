from .package_import import *



class lithologies():
    def __init__(self,Cp=1187,S=407,fr = 0.15,alpha = 3e-5,Ts = None,Tcpx=None,TLl=None,TL = None,MCpx = None,dTs = None, dTL = None, dTLl = None):
        self.Cp    = Cp
        self.alpha  = alpha 
        self.alpha_l = 6.8e-5
        self.rho   = 3300 
        self.liq   = 2900 
        self.S     = S 
        self.Ts    = Ts 
        self.dTs   = dTs
        self.TL    = TL
        self.dTL   = dTL  
        self.Tcpx  = Tcpx
        self.TLL   = TLl
        self.dTLL  = dTLl
        self.Mcpx   = MCpx
        self.beta = 1.50 
        self.r0    = 0.5
        self.r1    = 0.08/1e9
        self.fr    = fr
    
        
def dry_differentiation(L,a,b,Tp):
    
    dx = b * (L.alpha/L.Cp/L.rho) * np.exp((L.alpha/L.Cp/L.rho) * a)
    
    return dx         
 
def dry_adiabatic(L,a,b,Tp):
    
    T = Tp * np.exp((L.alpha/L.Cp/L.rho) * a)
    
    return T     

def dFdP_derivative(L,P,T,dP,F):
    
    dTdFP = dTdF_derivativeP(L,P,T,dP,F)
    dTdPF = dTdP_derivativeF(L,P,T,dP,F)
    
    a = -(L.Cp/T)*dTdPF
    alpha_l = (6.8e-5/L.liq)
    alpha_s = (L.alpha/L.rho)
    b = (L.S + (L.Cp/T) * dTdFP)
    
    dFdP = (a + F * alpha_l + (1 - F) * alpha_l)/(b) 
        
    return dFdP 

def dTdF_derivativeP(L,P,T,dP,F):
    
    if F > L.Mcpx(L.fr,P):

        
        
        a  = L.TL(P)-L.Tcpx(P) 
        F0 = L.Mcpx(P)
        n = (1-L.beta)/L.beta
        x = L.beta**(-1) * (a/(1-F0)) * ((F-F0)/(1-F0))**(n)  
        
    else: 
        a = L.TLL(P)-L.Ts(P)
        n = (1-L.beta)/L.beta
        x = (1/L.beta) * a * F ** n 
    
    
    return x 

def dTdP_derivativeF(L,P,T,dP,F):
        # Hell disgusting shit: 
        # ---- 
        # Equation:  
        # F     = F0 + (1-F0)*(T-Tc)/(Td-Tc))**(beta)
        # T     = ((F - F0)/(1-F0))**(1/beta) * (Td-Tc) + Tc 
        # dT/dP = ((F-F0(P))/(1-F0(P)))**(1/beta) * (Td(P)-Tc(P)) + Tc(P)
        # dT/dP = (A)**(1/b) * (B-C) +C 
        # dT/dP = [A'(A)**(1/b-1) * (B-C) + (B'-C')*A**(1/b) + C'  ]
        # C     = F0(P)**(1/b) * (Tb(P)-Ts(P)) + Ts(P)
        # C     = D**(1/b) * (E-F) + F 
        # C     = D'*D**(1/b-1)*(E-F) + D**(1/b)*(E'-F') + F' 
        # ---- 
        # -> This is Hell 
        
        
    a0 = L.Ts(P)
    b0 = L.TLL(P)
    c0 = L.Tcpx(L.fr,P)
    d0 = L.TL(P)
    
    
    a1 = L.dTs(P)  # dTdP solidus
    b1 = L.dTLL(P) # dTdP lheo 
    c1 = None      # dTdP T_cpx_out -> Hell     T = mcpx**(1/1.5) * (Tl-Ts) + Ts

    d1 = L.dTL(P)  # dTdP Tliquidus             
        
        
    if F > L.Mcpx(L.fr,P):

        
        
        """
            T    = ((F-F0)/(1-F0))**(1/beta) * (Tl-Tcpx) + Tcpx
            A(P) =  ((F-F0)/(1-F0))**(1/beta)
            
            T    = (A(P))*(g(P)-h(p)) + h(p)
            dTdP = (A'(P))*(g(P)-h(P)) + A(P)*(g'(P)-h'(P)) + h'(P)
            A'(P) = 1/beta *((1-F0)*(-dF0) - (-dF0)*(F-F0))/(1-F0)**2 * ((F-F0)/(1-F0))**(1/beta-1)
        
        
        
        """
        F0  = L.fr/(L.r0+L.r1 * P)
        dF0 = - ((L.fr*L.r1)/(L.r0+L.r1 * P)**2) 
        FF  = ((F-F0)/(1-F0))
        dFF = ((F-1)*dF0)/(F0-1)**2
        dfobeta = -(L.r1*(F0)**(1/L.beta))/(L.beta*(L.r1 * P + L.r0))
        
        
        c1 = (1/L.beta)*dF0*F0**(1/L.beta-1)*(d0-a0)+F0 ** (1/L.beta) * (d1-a1) + a1  
        #-((F - 1) * b * ϕ * ((F - ϕ / (b * x + a)) / (1 - ϕ / (b * x + a)))^(1 / β)) / (β * (b * x - ϕ + a) * (F * b * x - ϕ + F * a))
        A    = (1/L.beta) * dFF * (FF) ** (1/L.beta-1) 
        B    = (d0-c0) 
        B1   = (d1-c1) 
        dT = A * B + FF ** (1/L.beta) * B1 + c1 

    else: 
        dT = F**(1/L.beta) * (b1 - a1) + a1
    
    
    return dT 

def dTdPS(L,P,T,Tp,dP):
    
    F = compute_Fraction(T,L.TL(P),L.TLL(P),L.Ts(P),L.Tcpx(L.fr,P),L.Mcpx(L.fr,P))
    
    alpharho_l = 6.8e-5/L.liq
    alpharho_s = L.alpha/L.rho
    dFdP = dFdP_derivative(L,P,T,dP,F)
    
    dT = Tp * ((F * alpharho_l + (1-F) * alpharho_s  - L.S * dFdP)/L.Cp) 
    
    
    return dT 
    
def runge_kutta_algori(L,P,T,dP,F0,F1,Tp): 

    
    f = 1.0 
    if F0 == 0:
        dif = lambda P,T :dry_differentiation(L,P,  T, Tp)
        f=0.0
    else: 
        dif = lambda P,T :dTdPS(L,P,T,Tp,dP)

        # Calculate slopes
    k1 = dP * dif(P      , T     )
    k2 = dP * dif(P+dP/2., T+k1/2)
    k3 = dP * dif(P+dP/2., T+k2/2)
    k4 = dP * dif(P+dP   , T+k3  )

    # Calculate new x and y
    dX =  1./6*(k1+2*k2+2*k3+k4)

    return dX 
    
    
    
    

def SL(A,B,C,P):
    
    A = A+273.15 
    B = B/1e9
    C = C/(1e9)**2
    
    T = A + B * P + C * P**2 
        
    return T 

def dSL(B,C,P):
    
    B = B/1e9
    C = C/(1e9)**2
    
    dT = B  + 2 * C * P 
        
    return dT 

def rcpx(r0,r1,fr,P):
    
    r1 = r1/1e9
    
    Mcpx = fr/(r0+r1*P)
    
    return Mcpx 


def compute_T_cpx(mcpx,Ts,Tl):
    
    T = mcpx**(1/1.5) * (Tl-Ts) + Ts
    
    
    return T


def compute_Fraction(T,Tl,TlL,Ts,Tcpx,Mcpx):
    
    if T < Ts: 
        F = 0
    else:
        F = ((T-Ts)/(TlL-Ts))**(1.50)
    if F > Mcpx: 
        
        F = Mcpx+(1-Mcpx) * ((T-Tcpx)/(Tl-Tcpx))**(1.50) 
    
    
    return F


def function_test():

    P    = np.linspace(0,5.0e9,num=1000)

    T    = np.linspace(1000.15,1973.15,num=100)

    TS       = lambda  P :SL(1085.7, 132.9,-5.1,P)
    
    dTS      = lambda  P :dSL(132.9,-5.1,P)

    TLL      = lambda  P :SL(1475,80,-3.2,P)
    
    dTLL     = lambda P  :dSL(80,-3.2,P)

    TL       = lambda  P :SL(1780,45.0,-2.0,P)
    
    dTL      = lambda  P :dSL(45.0,-2.0,P)

    Mcpx     = lambda  P :rcpx(0.5,0.08,0.10,P)

    TCpx     = lambda  P :compute_T_cpx(Mcpx(P),TS(P),TLL(P))

    Tspx     = lambda  P :SL(1095,124.1,-4.7,P) 

    Tspx_cpx = lambda  P :SL(1179.6,157.2,-11.1,P)

    fig = plt.figure()

    plt.plot(TS(P) - 273.15 , P ,linestyle='-.', c = 'k'          , linewidth = 0.8)
    plt.plot(TLL(P) - 273.15, P ,linestyle='--', c = 'forestgreen', linewidth = 0.8)
    plt.plot(TL(P)  - 273.15, P ,linestyle='-.', c = 'firebrick'  , linewidth = 0.8)
    plt.plot(TCpx(P)- 273.15, P ,linestyle='--', c = 'b'          , linewidth = 0.8)




    P_exp = [0,1e9,2e9,3e9] 
    F0 = np.zeros(len(T))
    F1 = np.zeros(len(T))
    F2 = np.zeros(len(T))
    F3 = np.zeros(len(T))
    for i in range(len(T)): 
        F0[i] = compute_Fraction(T[i],TL(P_exp[0]),TLL(P_exp[0]),TS(P_exp[0]),TCpx(P_exp[0]),Mcpx(P_exp[0]))
        F1[i] = compute_Fraction(T[i],TL(P_exp[1]),TLL(P_exp[1]),TS(P_exp[1]),TCpx(P_exp[1]),Mcpx(P_exp[1]))
        F2[i] = compute_Fraction(T[i],TL(P_exp[2]),TLL(P_exp[2]),TS(P_exp[2]),TCpx(P_exp[2]),Mcpx(P_exp[2]))
        F3[i] = compute_Fraction(T[i],TL(P_exp[3]),TLL(P_exp[3]),TS(P_exp[3]),TCpx(P_exp[3]),Mcpx(P_exp[3]))


    fig = plt.figure()


    plt.plot(T-273.15, F0,c='k',          linestyle = ':', linewidth = 0.9)

    plt.plot(T-273.15, F1,c='r',          linestyle = ':', linewidth = 0.9)

    plt.plot(T-273.15, F2,c='forestgreen',linestyle = ':', linewidth = 0.9)

    plt.plot(T-273.15, F3,c='blue',       linestyle = ':', linewidth = 0.9)


    # ------ Oliver Shorttle approach ----- # 

    lhz = lithologies(1000,300,TS,TCpx,TLL,TL,Mcpx,dTS,dTL,dTLL)
    pyr = lithologies(1140,380)
    har = lithologies(1000,0.0)


    # Again a matrioska paper referencing. Richard uses a lhz for doing his computation, claiming to have followed the parametrisation of Schorle, but schorle uses a composite
    # litholigies made of Harzburgite, pyroxenite and lhz. Lhz is coming from Katz 2003, where, surprise surprise you find the adiabatic contribution. I will use this formulation for 
    # making my life easier, otherwise, becomes an incredible mess to understand the paper of Schorle, whose notation seems the wildest dream of Azazoth. 

    P       = np.linspace(0e9,5e9,num=2000)
    T       = np.zeros(len(P))
    P       = np.flip(P)
    F       = np.zeros(len(P))
    Tp = 1450 + 273.15
    T_start = (Tp) + 18*5.0
    T[0] = T_start
    fig = plt.figure()
    ax = fig.gca()
    
    for i in range(len(P)-1):
        dP = P[i+1]- P[i]
        F0  = compute_Fraction(T[i],TL(P[i]),TLL(P[i]),TS(P[i]),TCpx(P[i]),Mcpx(P[i]))  
        dT = runge_kutta_algori(lhz,P[i],T[i],dP,F0,None,Tp)
        T[i+1] = T[i] + dT 
        if F0>0.0 and F0 <= lhz.Mcpx(P[i]):
            c = 'r'
            alpha = 0.8 
        elif F0 > lhz.Mcpx(P[i]):
            c = 'forestgreen'
            alpha = 0.9
        else:
            c='k'
            alpha = 0.3
        F[i]   = F0
        
        ax.scatter(T[i]-273.15,P[i],c=c,s=0.5,alpha = alpha)   
        
    ax.invert_yaxis()
    
    ax.plot(TS(P) - 273.15 , P ,linestyle='-.', c = 'k'          , linewidth = 0.8)
    ax.plot(TLL(P) - 273.15, P ,linestyle='--', c = 'forestgreen', linewidth = 0.8)
    ax.plot(TL(P)  - 273.15, P ,linestyle='-.', c = 'firebrick'  , linewidth = 0.8)
    ax.plot(TCpx(P)- 273.15, P ,linestyle='--', c = 'b'          , linewidth = 0.8)
    
    
    
    
    F[-1] = compute_Fraction(T[-1],TL(P[-1]),TLL(P[-1]),TS(P[-1]),TCpx(P[-1]),Mcpx(P[-1]))  
    fig = plt.figure()
    ax = fig.gca()
    plt.plot(F[F>0],P[F>0]/1e9)
    ax.invert_yaxis()
        
    print ('bla')




    
if __name__ == '__main__': 
    
    function_test()
 
