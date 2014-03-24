import numpy as np
import openopt as op


def so_sol(adj, edge_list, a, b, d):
    """
    Solves the system optimal traffic assignment problem for 1 O-D pair.
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
    bb = np.zeros(len(edge_list))
    
    Aeq = np.zeros(nodes*edges).reshape(nodes,edges)
    beq = np.zeros(nodes)
    
    beq[0] = -d  # it is assumed that origin node only has outgoing edges
    beq[-1] = d  # it is assumed that destination node only has incoming edges
    
    for n in range(nodes):
        
        incoming = adj[:,n]
        outgoing = adj[n,:]
        
        node_index_in = [i for i in range(len(incoming)) if incoming[i]!=0]
        node_index_out = [i for i in range(len(outgoing)) if outgoing[i]!=0]
        
        incoming_edges = [edge_list.index((i,n)) for i in node_index_in] 
        outgoing_edges = [edge_list.index((n,i)) for i in node_index_out]
        
        for i in incoming_edges:
            Aeq[n,i] = 1
        for i in outgoing_edges:
            Aeq[n,i] = -1
    
    
    op.QP()
    
    p = op.QP(H = H, f = f, A = A, b = bb, Aeq = Aeq, beq = beq)
    r = p._solve('cvxopt_qp', iprint = 0)
    f_opt, x_opt = r.ff, r.xf
    
    return x_opt
    
    
def ue_sol(adj, edge_list, a, b, d):
    """
    Solves the system optimal traffic assignment problem for 1 O-D pair.
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
    
    H = np.diagflat(b)
    f = a
    
    A = -np.eye(len(edge_list))
    bb = np.zeros(len(edge_list))
    
    Aeq = np.zeros(nodes*edges).reshape(nodes,edges)
    beq = np.zeros(nodes)
    
    beq[0] = -d  # it is assumed that origin node only has outgoing edges
    beq[-1] = d  # it is assumed that destination node only has incoming edges
    
    for n in range(nodes):
        
        incoming = adj[:,n]
        outgoing = adj[n,:]
        
        node_index_in = [i for i in range(len(incoming)) if incoming[i]!=0]
        node_index_out = [i for i in range(len(outgoing)) if outgoing[i]!=0]
        
        incoming_edges = [edge_list.index((i,n)) for i in node_index_in] 
        outgoing_edges = [edge_list.index((n,i)) for i in node_index_out]
        
        for i in incoming_edges:
            Aeq[n,i] = 1
        for i in outgoing_edges:
            Aeq[n,i] = -1
    
    
    op.QP()
    
    p = op.QP(H = H, f = f, A = A, b = bb, Aeq = Aeq, beq = beq)
    r = p._solve('cvxopt_qp', iprint = 0)
    f_opt, x_opt = r.ff, r.xf
    
    return x_opt