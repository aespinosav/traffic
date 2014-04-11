from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def F_SO(d, a, b, am, bm):
    
    d_on = (1- am - a)/(2*(1+bm))
    d_off = (1- am - a)/(1-b)
    
    if d < d_on:
        cost = (2+bm)*d**2 + (2*a+am)*d
    elif d < d_off:
        cost = 0.5*(1+b-(((1-b)**2)/(1+2*bm+b)))*d**2 + (1+a -((1-b)*(1- am - a))/(1+2*bm+b))*d - ((1-am-a)**2)/(1+2*bm+b)
    else:
        cost = ((1+b)/(2))*d**2 + (a+1)*d
        
    return cost
    
#def F_SO_alt(d, a, b, am, bm):
    
    #c = []
    
    #c1 = (2+bm)*d**2 + (2*a+am)*d
    #c2 = 0.5*(1+b-((1-b)**2/(1+2*bm+b)))*d**2 + (1+a -((1-b)*(1- am - a))/(1+2*bm+b))*d - (1-am-a)**2/(1+2*bm+b)
    #c3 = ((1+b)/(2))*d**2 + (a+1)*d
    
    #if c1 >= 0:
        #c.append(c1)
    #if c2 >= 0:
        #c.append(c2)
    #if c3 >= 0:
        #c.append(c3)
    
    #return min(c)
    
def F_SO_A(d, a, b, am, bm):
    
    cost = (2+bm)*d**2 + (2*a+am)*d
    return cost
    
def F_SO_B(d, a, b, am, bm):
    cost = 0.5*(1+b-((1-b)**2/(1+2*bm+b)))*d**2 + (1+a -((1-b)*(1- am - a))/(1+2*bm+b))*d - (1-am-a)**2/(1+2*bm+b)
    return cost
    
def F_SO_C(d, a, b, am, bm):
    cost = ((1+b)/(2))*d**2 + (a+1)*d
    return cost
    

    
def F_UE(d, a, b, am, bm):
    
    d_on = (1- am - a)/(1+bm)
    d_off = 2*(1- am - a)/(1-b)
    
    if d < d_on:
        cost = (2+bm)*d**2 + (2*a+am)*d
    elif d < d_off:
        cost = 0.5*(1+b-((1-b)**2/(1+2*bm+b)))*d**2 + (1+a -((1-b)*(1- am - a))/(1+2*bm+b))*d
    else:
        cost = ((1+b)/2)*d**2 + (a+1)*d
        
    return cost

    
    
a = 0.5 
b = 0.5
am= 0.1
bm= 0.1

D = np.arange(0, 1.5*2*(1- am - a)/(1-b) + 0.001, 0.001)

#ca = []
#cb = []
#cc = []

#for d in D:
    #ca.append(F_SO_A(d, a, b, am, bm))
    #cb.append(F_SO_B(d, a, b, am, bm))
    #cc.append(F_SO_C(d, a, b, am, bm))
    
#plt.plot(D, ca, "g-", label="A")
#plt.plot(D, cb, "y-", label="B")
#plt.plot(D, cc, "r-", label="C")











SO_cost =[]
#SO_cost_alt =[]
UE_cost =[]
for d in D:
    SO_cost.append(F_SO(d, a, b, am, bm))
    #SO_cost_alt.append(F_SO_alt(d, a, b, am, bm))
    UE_cost.append(F_UE(d, a, b, am, bm))

plt.plot(D, SO_cost, "g-", label="SO")
#plt.plot(D, SO_cost_alt, "m--", label="SO_min")
plt.plot(D, UE_cost, "r-", label="UE")

#plt.axvline(x=(1- am - a)/(1+bm), ls="b--")

plt.xlabel("Demand")
plt.ylabel("Total Cost")

plt.legend()