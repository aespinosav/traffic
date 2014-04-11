import numpy as np
import matplotlib.pyplot as plt

from testing_values import *
from ta_solver import *



UE_sols = ta_range_solve(D, adj, edge_list, a, b, "UE")
SO_sols = ta_range_solve(D, adj, edge_list, a, b, "SO")

SO_costs = total_cost_func(SO_sols, a, b)
UE_costs = total_cost_func(UE_sols, a, b)

#for i in range(len(D)):
    #SO_costs.append(total_cost(SO_sols[i], a, b))
    #UE_costs.append(total_cost(UE_sols[i], a, b))


plt.plot(D, SO_costs, "g", label="SO")
plt.plot(D, UE_costs, "r", label="UE")

plt.axvline(x=d_onUE,  color="r", linestyle="--")
plt.axvline(x=d_offUE,  color="r", linestyle="--")

plt.axvline(x=d_onSO,  color="g", linestyle="--")
plt.axvline(x=d_offSO,  color="g", linestyle="--")

plt.xlabel("Demand")
plt.ylabel("Total Cost")
plt.title("Total costs for UE and SO assignment")

plt.xlim([0,1.7])
plt.ylim([0,5])

plt.legend(loc=2)