import numpy as np

aa = 0.5
bb = 0.5
am = 0.1
bm = 0.1

adj = np.array([0,1,1,0,0,0,1,1,0,0,0,1,0,0,0,0]).reshape(4,4)
edge_list = edge_list = [(0,1),(0,2),(2,3),(1,3),(1,2)]

a = np.array([aa, 1, aa, 1, am])
b = np.array([1, bb, 1, bb, bm])

d_onUE = (1.0-am-aa)/(1+bm)
d_offUE = 2*(1.0-am-aa)/(1- bb)

d_onSO = 0.5*(1.0-am-aa)/(1+bm)
d_offSO = (1.0-am-aa)/(1- bb)

D = np.arange(0, 3.001, 0.001)