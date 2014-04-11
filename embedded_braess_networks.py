import numpy as np

"""
This file contains adjacency matrices for the embedded
braess networks (see stuff in binder), hopefully a pdf
with the drawing for each will be added soon.
"""

adj_braess = np.array([0,1,1,0,
                       0,0,1,1,
                       0,0,0,1,
                       0,0,0,0]).reshape(4,4)


#Adjacent braess networks:
#down:

adj_ad = np.array([0,1,1,1,0,
                   0,0,1,0,1,
                   0,0,0,1,1,
                   0,0,0,0,1,
                   0,0,0,0,0]).reshape(5,5)

#inwards                   
adj_ai = np.array([0,1,1,1,0,
                   0,0,0,1,1,
                   0,0,0,1,1,
                   0,0,0,0,1,
                   0,0,0,0,0]).reshape(5,5)


#Embedded braess networks:
#in link 1 (case A):

adj_e1 = np.array([0,1,1,0,1,0,
                   0,0,1,1,0,0,
                   0,0,0,1,0,0,
                   0,0,0,0,1,1,
                   0,0,0,0,0,1,
                   0,0,0,0,0,0]).reshape(6,6)
                   
#in link 4 (case B):

adj_e2 = np.array([0,1,1,0,0,0,
                   0,0,1,1,0,0,
                   0,0,0,1,0,0,
                   0,0,0,0,1,1,
                   0,0,0,0,0,1,
                   0,0,0,0,0,0]).reshape(6,6)

                   
#in link 5 (case C) check order of nodes in binder to keep matrix upper triangular

adj_e3 = np.array([0,1,0,0,1,0,
                   0,0,1,1,0,1,
                   0,0,0,1,1,0,
                   0,0,0,0,1,0,
                   0,0,0,0,0,1,
                   0,0,0,0,0,0]).reshape(6,6)