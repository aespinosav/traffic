import numpy as np
import openopt as op


def ue_sol(adj, edge_list, a, b, d):
    """
    Solves the user equilibrium traffic assignment problem for 1 O-D pair.
    for linear cost functions of the type f = a + bx.
    
    adj - adjacency matrix of traffic network
    edge_list - ordered edge list
    a - array of "a" parameters of link cost functions (order according to edge_list)
    b - array of "b" parameters of link cost functions (order according to edge_list)
    
    d- demand on the network
    
    node 0 should be the origin
    the last node should be the destination
    """
    nodes = len(adj)
    edges = len(edge_list)
    
    H = np.diagflat(2*b)
    f = a
    
    A = -np.eye(len(edge_list))
    bb = zeros(len(edge_list))
    
    Aeq = np.zeros(nodes*edges).reshape(nodes,edges)
    
    for n in range(nodes):
        
        incoming = adj[:,n]
        outgoing = adj[n,:]
        
        node_index_in = [i for i in range(len(incoming)) if incoming[i]!=0]
        node_index_out = [i for i in range(len(outgoing)) if outgoing[i]!=0]
        
        incoming_edges = [edge_list.index((n,i)) for i in node_index_in] 
        outgoing_edges = [edge_list.index((i,n)) for i in node_index_out]
        
        

    
    
    
    
    
    
    
    
    op.QP()