#from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


#User equilibrium
def y3_UEa(d, a, b, am, bm):
    return d

def y3_UEb(d, a, b, am, bm):
    
    y3 = 2*(1.0-am-a)/(1+2*bm+b) - ((1.0-b)/(1+2*bm+b))*d
    return y3
    
def y3_UEc(d, a, b, am, bm):
    return 0*d
    
def y1_UEa(d, a, b, am, bm):
    return 0*d
    
def y1_UEb(d, a, b, am, bm):
    return (d - y3_UEb(d, a, b, am, bm))/2.0
    
def y1_UEc(d, a, b, am, bm):
    return d/2.0
    



#SO
def y3_SOa(d, a, b, am, bm):
    return d
    
def y3_SOb(d, a, b, am, bm):
    
    y3 = (1.0-am-a)/(1.0+2*bm+b) -((1.0-b)/(1+2*bm+b))*d
    return y3    
    
def y3_SOc(d, a, b, am, bm):
    return d*0

def y1_SOa(d, a, b, am, bm):
    return 0*d
    
def y1_SOb(d, a, b, am, bm):
    return (d - y3_SOb(d, a, b, am, bm))/2.0
    
def y1_SOc(d, a, b, am, bm):
    return d/2.0    
    
    
    
a = 0.5 
b = 0.5
am= 0.1
bm= 0.1
    
D = np.arange(0, 3, 0.001)

d_onUE = (1.0-am-a)/(1+bm)
d_offUE = 2*(1.0-am-a)/(1- b)

d_onSO = 0.5*(1.0-am-a)/(1+bm)
d_offSO = (1.0-am-a)/(1- b)




##UE
#y3_UE = []

#for d in D:
    #if d < d_onUE:
        #y3_UE.append(y3_UEa(d, a, b, am, bm))
    #elif d < d_offUE:
        #y3_UE.append(y3_UEb(d, a, b, am, bm))
    #else:
        #y3_UE.append(y3_UEc(d, a, b, am, bm))
        
#y1_UE = []

#for d in D:
    #if d < d_onUE:
        #y1_UE.append(y1_UEa(d, a, b, am, bm))
    #elif d < d_offUE:
        #y1_UE.append(y1_UEb(d, a, b, am, bm))
    #else:
        #y1_UE.append(y1_UEc(d, a, b, am, bm))

        
#plt.plot(D, y3_UE, "b-", label="Inner flow")
#plt.plot(D, y1_UE, "g-", label="Outer flow")
#plt.axvline(x=d_onUE,  color="k", linestyle="--")
#plt.axvline(x=d_offUE,  color="k", linestyle="--")

#plt.xlabel("Demand")
#plt.ylabel("Flow")
#plt.xlim([0,3])

#plt.legend(loc=4)


#SO

y3_SO = []

for d in D:
    if d < d_onSO:
        y3_SO.append(y3_SOa(d, a, b, am, bm))
    elif d < d_offSO:
        y3_SO.append(y3_SOb(d, a, b, am, bm))
    else:
        y3_SO.append(y3_SOc(d, a, b, am, bm))
        
y1_SO = []

for d in D:
    if d < d_onSO:
        y1_SO.append(y1_SOa(d, a, b, am, bm))
    elif d < d_offSO:
        y1_SO.append(y1_SOb(d, a, b, am, bm))
    else:
        y1_SO.append(y1_SOc(d, a, b, am, bm))

        
plt.plot(D, y3_SO, "b-", label="Inner flow")
plt.plot(D, y1_SO, "g-", label="Outer flow")

plt.axvline(x=d_onSO,  color="k", linestyle="--")
plt.axvline(x=d_offSO,  color="k", linestyle="--")

plt.xlabel("Demand")
plt.ylabel("Flow")
plt.xlim([0,3])
plt.legend(loc=4)