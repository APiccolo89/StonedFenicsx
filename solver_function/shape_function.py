
import numpy as np

from mpmath import mpf, mpc, mp

mp.dfs = 60

#------------------------------------------------------------------------------
# Crouzeix-Raviart elements for Stokes, P_2 for Temp.
# the internal numbering of the nodes is as follows:
#
#  P_2^+            P_-1             P_2 
#
#  02           #   02           #   02
#  ||\\         #   ||\\         #   ||\\
#  || \\        #   || \\        #   || \\
#  ||  \\       #   ||  \\       #   ||  \\
#  04   03      #   ||   \\      #   04   03
#  || 06 \\     #   ||    \\     #   ||    \\
#  ||     \\    #   ||     \\    #   ||     \\
#  00==05==01   #   00======01   #   00==05==01


# Shape function for the velocity field: second order triangle element with barycentral node

def NN_t_P_2_7(r,s):
    r = mpf(r)
    s = mpf(s)

    NV_0= (1-r-s)*(1.0-2.0*r-2.0*s+ 3.0*r*s)
    NV_1= r*(2.0*r -1.0 + 3.0*s-3.0*r*s-3.0*s**2 )
    NV_2= s*(2.0*s -1 + 3.0*r-3.0*r**2-3.0*r*s )
    NV_3= 4.0*(1.0-r-s)*r*(1.0-3.0*s)
    NV_4= 4.0*r*s*(-2.0+3.0*r+3.0*s)
    NV_5= 4.0*(1.0-r-s)*s*(1.0-3.0*r)
    NV_6= 27.0*(1.0-r-s)*r*s

    return np.float64(NV_0),np.float64(NV_1),np.float64(NV_2),np.float64(NV_3),np.float64(NV_4),np.float64(NV_5),np.float64(NV_6)


def dNN_dr_t_P_2_7(r,s):
    r = mpf(r)
    s = mpf(s)

    dNdr_0= r*(4.0-6.0*s)-3.0*s**2+7.0*s-3.0
    dNdr_1= r*(4.0-6.0*s)-3.0*s**2+3.0*s-1.0
    dNdr_2= -3.0*s*(2.0*r+s-1.0)  
    dNdr_3= 4.0*(3.0*s-1.0)*(2.0*r+s-1.0) 
    dNdr_4= 4.0*s*(6.0*r+3.0*s-2.0) 
    dNdr_5= 4.0*s*(6.0*r+3.0*s-4.0)
    dNdr_6=-27.0*s*(2.0*r+s-1.0)

    return np.float64(dNdr_0),np.float64(dNdr_1),np.float64(dNdr_2),np.float64(dNdr_3),np.float64(dNdr_4),np.float64(dNdr_5),np.float64(dNdr_6)

def dNN_ds_t_P_2_7(r,s):
    r = mpf(r)
    s = mpf(s)

    dNds_0= -3.0*r**2+r*(7.0-6.0*s)+4.0*s-3.0
    dNds_1= -3.0*r*(r+2.0*s-1.0)
    dNds_2= -3.0*r**2+r*(3.0-6.0*s)+4.0*s-1 
    dNds_3= 4.0*r*(3.0*r+6.0*s-4.0)  
    dNds_4= 4.0*r*(3.0*r+6.0*s-2.0) 
    dNds_5= 4.0*(3.0*r-1.0)*(r+2.0*s-1)
    dNds_6= -27.0*r*(r+2.0*s-1)
    return np.float64(dNds_0),np.float64(dNds_1),np.float64(dNds_2),np.float64(dNds_3),np.float64(dNds_4),np.float64(dNds_5),np.float64(dNds_6)


# Shape function for the pressure field: DISCONTINUOUS linear triangle element

def NN_t_P_1_3(rq,sq):
    rq = mpf(rq)
    sq = mpf(sq)

    NP_0=1.-rq-sq
    NP_1=rq
    NP_2=sq
    return np.float64(NP_0),np.float64(NP_1),np.float64(NP_2)

def NN_t_P_1_3(rq,sq):
    rq = mpf(rq)
    sq = mpf(sq)

    NP_0=1.-rq-sq
    NP_1=rq
    NP_2=sq
    return np.float64(NP_0),np.float64(NP_1),np.float64(NP_2)

#------------------------------------------------------------------------------
# # Shape function for the velocity field: second order triangle element
#------------------------------------------------------------------------------

def NN_t_P_2_6(r,s):
    r = mpf(r)
    s = mpf(s)

    N_0=1-3*r-3*s+2*r**2+4*r*s+2*s**2
    N_1=-r+2*r**2
    N_2=-s+2*s**2
    N_3=4*r-4*r**2-4*r*s
    N_4=4*r*s
    N_5=4*s-4*r*s-4*s**2

    return np.float64(N_0),np.float64(N_1),np.float64(N_2),np.float64(N_3),np.float64(N_4),np.float64(N_5)

def dNN_dr_t_P_2_6(r,s):

    r = mpf(r)
    s = mpf(s)

    dNdr_0=-3+4*r+4*s
    dNdr_1=-1+4*r
    dNdr_2=0
    dNdr_3=4-8*r-4*s
    dNdr_4=4*s
    dNdr_5=-4*s
    return np.float64(dNdr_0),np.float64(dNdr_1),np.float64(dNdr_2),np.float64(dNdr_3),np.float64(dNdr_4),np.float64(dNdr_5)

def dNN_ds_t_P_2_6(r,s):
    r = mpf(r)
    s = mpf(s)

    dNds_0=-3+4*r+4*s
    dNds_1=0
    dNds_2=-1+4*s
    dNds_3=-4*r
    dNds_4=4*r
    dNds_5=4-4*r-8*s
    return np.float64(dNds_0),np.float64(dNds_1),np.float64(dNds_2),np.float64(dNds_3),np.float64(dNds_4),np.float64(dNds_5)

