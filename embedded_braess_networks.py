import numpy as np
import networkx as nx

"""
This file contains adjacency matrices for the embedded
braess networks (see stuff in binder), hopefully a pdf
with the drawing for each will be added soon.
"""


a = 0.5
b = 0.5
am = 0.1
bm = 0.1

adj_braess = np.array([0,1,1,0,
                       0,0,1,1,
                       0,0,0,1,
                       0,0,0,0]).reshape(4,4)

a_coefs = [a, 1, am, 1, a]
b_coefs = [1, b, bm, b, 1]
                       
                       

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

#a_coefs = [a, a, 1, am, 1, am, 1, a]
#b_coefs = [1, 1, b, bm, b, bm, b, 1]


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
                   0,0,1,1,1,0,
                   0,0,0,0,0,1,
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
                   
def make_network(adj, a_coefs, b_coefs):
    """
    Makes directed network from adjacency matrix and deletes the 'weight' attr.
    
    It also gives the edges their a and b coef as attributes
    """
    
    g = nx.DiGraph(adj)
    
    for i in range(len(g.edges())):
        e = g.edges()[i]
        del g[e[0]][e[1]]['weight']
        g[e[0]][e[1]]['a'] = a_coefs[i]
        g[e[0]][e[1]]['b'] = b_coefs[i]
    
    return g