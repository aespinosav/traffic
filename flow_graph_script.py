import numpy as np
import matplotlib.pyplot as plt
from ta_solver import *
from flows import *

import matplotlib as mpl


aa = 0.5 
bb = 0.5
am= 0.1
bm= 0.1
    
D = np.arange(0, 3, 0.001)

d_onUE = (1.0-am-aa)/(1+bm)
d_offUE = 2*(1.0-am-aa)/(1- bb)

d_onSO = 0.5*(1.0-am-aa)/(1+bm)
d_offSO = (1.0-am-aa)/(1- bb)

adj = np.array([0,1,1,0,0,0,1,1,0,0,0,1,0,0,0,0]).reshape(4,4)
edge_list = edge_list = [(0,1),(0,2),(2,3),(1,3),(1,2)]

a = np.array([aa, 1, aa, 1, am])
b = np.array([1, bb, 1, bb, bm])



so_numerical_sols = []
ue_numerical_sols = []

for d in D:
    so_sol = ta_solve(adj, edge_list, a, b, d, "SO")
    so_numerical_sols.append(so_sol)
    
    ue_sol = ta_solve(adj, edge_list, a, b, d, "UE")
    ue_numerical_sols.append(ue_sol)
    
y1_num_so = [x[4] for x in so_numerical_sols]
y3_num_so = [x[0] - x[4] for x in so_numerical_sols]

y1_num_ue = [x[4] for x in ue_numerical_sols]
y3_num_ue = [x[0] - x[4] for x in ue_numerical_sols]



#UE
y3_UE = []
y1_UE = []


for d in D:
    if d < d_onUE:
        y3_UE.append(y3_UEa(d, aa, bb, am, bm))
        y1_UE.append(y1_UEa(d, aa, bb, am, bm))
    elif d < d_offUE:
        y3_UE.append(y3_UEb(d, aa, bb, am, bm))
        y1_UE.append(y1_UEb(d, aa, bb, am, bm))
    else:
        y3_UE.append(y3_UEc(d, aa, bb, am, bm))
        y1_UE.append(y1_UEc(d, aa, bb, am, bm))

#SO

y3_SO = []
y1_SO = []

for d in D:
    if d < d_onSO:
        y3_SO.append(y3_SOa(d, aa, bb, am, bm))
        y1_SO.append(y1_SOa(d, aa, bb, am, bm))
    elif d < d_offSO:
        y3_SO.append(y3_SOb(d, aa, bb, am, bm))
        y1_SO.append(y1_SOb(d, aa, bb, am, bm))
    else:
        y3_SO.append(y3_SOc(d, aa, bb, am, bm))
        y1_SO.append(y1_SOc(d, aa, bb, am, bm))
    
    
    
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['font.sans-serif'] = 'Helvetica' 


fig1 = plt.figure()
ax = fig1.add_subplot(111)

ax.plot(D, y3_SO, "r-", label="Inner flow ($y_3$)")
ax.plot(D, y1_SO, "g-", label="Outer flow ($y_1$)")

ax.plot(D, y3_num_so, "k--")
ax.plot(D, y1_num_so, "k--")

ax.axvline(x=d_onSO,  color="k", linestyle="--")
ax.axvline(x=d_offSO,  color="k", linestyle="--")

ax.set_title("System Optimal")
ax.set_xlabel("Demand")
ax.set_ylabel("Flow")
ax.set_xlim([0,3])
ax.set_ylim([0,1.6])
ax.legend(loc=4)

#fig1.savefig("SO_flows.pdf")


#plt.clf()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

ax2.plot(D, y3_UE, "r-", label="Inner flow ($y_3$)")
ax2.plot(D, y1_UE, "g-", label="Outer flow ($y_1$)")

ax2.plot(D, y3_num_ue, "k--")
ax2.plot(D, y1_num_ue, "k--")

ax2.axvline(x=d_onUE,  color="k", linestyle="--")
ax2.axvline(x=d_offUE,  color="k", linestyle="--")

ax2.set_title("User Equilibrium")
ax2.set_xlabel("Demand")
ax2.set_ylabel("Flow")
ax2.set_xlim([0,3])
ax2.set_ylim([0,1.6])
ax2.legend(loc=4)

#fig2.savefig("UE_flows.pdf")
