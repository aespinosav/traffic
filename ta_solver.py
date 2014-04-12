from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import networkx as nx

def ta_solve(adj, edge_list, a_coefs, b_coefs, d, regime="SO"):
    """
    Solves the traffic assignment problem for 1 O-D pair.
    for linear cost functions of the type f = a + bx.
    
    Solves for either UE or SO passed as a string as the regime argument

    adj - adjacency matrix of traffic network
    edge_list - ordered edge list (edges represented by tuples of nodes eg. (0,1) )
    a - array of "a" parameters of link cost functions (order according to edge_list)
    b - array of "b" parameters of link cost functions (order according to edge_list)
    
    d - demand on the network
    
    node 0 should be the origin
    the last node should be the destination
    """
    
    nodes = len(adj)
    edges = len(edge_list)
    
    if regime == "SO":
        Q = 2*np.diag(b_coefs)
    elif regime == "UE":
        Q = np.diag(b_coefs)
    else:
        print "ERROR: regime must be either 'UE' or 'SO'"
        
    Q = matrix(Q)
    p = matrix(a_coefs)
    
    G = -np.eye(edges)
    G = matrix(G)
    
    h = np.zeros(edges)
    h = matrix(h)
    
    A = np.zeros(nodes*edges).reshape(nodes, edges)

    for i in range(len(adj)):

        outgoing = adj[i,:]
        incoming = adj[:,i]
        
        outgoing_edges = []
        incoming_edges = []     
        
        for j in range(len(outgoing)):
            
            if outgoing[j] != 0:
                out_edge = (i, j)
                outgoing_edges.append(out_edge)
                
            if incoming[j] != 0:
                in_edge = (j,i)
                incoming_edges.append(in_edge)

        
        incoming_variable_list = [edge_list.index(k) for k in incoming_edges]
        outgoing_variable_list = [edge_list.index(k) for k in outgoing_edges]
        
        #no self loops allowed
        for k in incoming_variable_list:
                A[i, k] = 1.0
        for k in outgoing_variable_list:
                A[i, k] = -1.0  
    
    b = np.zeros(nodes)
    b[0] = -d
    b[-1] = d
    
    augmented_matrix = np.column_stack([A, b])    
    
    #We use sympy to obtain the matrix in row reduced echelon form
    #This is to do it exactly and avoid numerical errors (might make the code really slow though...)	
    AM = sp.Matrix(augmented_matrix)
    AM = np.array( AM.rref()[0] )
    
    new_AM = []
    rows = 0
    for row in AM:
        if any(row):
            new_AM.append(row)
            rows += 1
            
    AM = np.array(new_AM).astype(float)
    
    A = AM[:,:-1].copy()
    A = A.transpose().reshape(len(A)*len(A[0]),)
    A = matrix(A, (rows, edges))
    
    b = AM[:,-1].copy()
    b = matrix(b)
    
    sol = solvers.qp(Q, p, G, h, A, b)
    
    x = sol["x"]
    x = np.array([i for i in x])
    
    
    return x
    
def ta_range_solve(D, adj, edge_list, a_coefs, b_coefs, regime="SO"):
    """
    Solves the traffic assignment for a range of demand using ta_solve.
    """
    
    sols = []
    for d in D:
        x = ta_solve(adj, edge_list, a_coefs, b_coefs, d, regime)
        sols.append(x)
        
    return np.array(sols)
    
    
def total_cost(x, a, b):
    """
    Calculates the total cost for a given demand on the network
    with linear cost functions.
    
    x - vector of link flows
    a - vector of constant terms in cost functions
    b - vector of linear coefficients of cost funtions
    
    cost functions allowed are of the form f(x) = a + bx.
    """
    
    x = np.array(x)
    a = np.array(x)
    b = np.array(x)
    
    cost = np.dot(x,a) + np.dot(x, b*x)
    
    return cost
    
def total_cost_func(X, a, b):
    """
    Calculates the total cost for an array of flow assignment vectors.
    to make graphs of cost changing with demand. Therefore, X should be
    an array of traffic flows for a specific demand value
    """
    
    costs = [total_cost(x, a, b) for x in X]
    costs = np.array(costs)
    
    return costs
    

def ta_solve_network(g, demand, regime):
    """Solves the traffic assignment problem for a network.
    
    g - networkx network object
    demand - a demand range for which to solve the ta problem
    """
    
    adj = np.array(nx.adjacency_matrix(g))
    edge_list = g.edges()
    
    a_coefs = [g[e[0]][e[1]]['a'] for e in g.edges_iter()]
    b_coefs = [g[e[0]][e[1]]['b'] for e in g.edges_iter()]
    
    sols = ta_range_solve(demand, adj, edge_list, a_coefs, b_coefs, regime)
    
    return sols
    
    
def plot_graph_for_flows(g, OD=None):
    
    pos = nx.spring_layout(g)
    
    #Make dictionary for edge labels
    edge_labels = []
    for i in range(len(g.edges())):
        u, v = g.edges()[i]
        edge_labels.append(((u,v,),i+1))
    edge_labels = dict(edge_labels)
    
    if OD is None:
        origin = g.nodes()[0]
        destination = g.nodes()[-1]
    else:
        origin, destination = OD
        
    node_label_dict = {origin:'O', destination:'D'}
    
    nx.draw_networkx(g, pos, labels=node_label_dict)
    nx.draw_networkx_edge_labels(g,pos,edge_labels=edge_labels, font_size=14)
    plt.axis('equal')
    plt.axis('off')

    
def plot_flows(D, sols, legend=False):
    """
    Plots the flows obtained from ta_range_solve when given the demand range and the flow solutions
    """
    
    for i in range(len(sols[0])):
        plt.plot(D, sols[:,i], label="Flow {}".format(i+1))
    
    if legend:
        plt.legend(loc=4)
        
    plt.xlabel('Demand')
    plt.ylabel('Flow')
    
    #todo: plot the graph here as well (possibly in upper left corner) / maybe if graph is smaller than N nodes
        
    plt.show()
    
    

##Testing values (this should be commented out when using as a library)
#aa = 0.5
#bb = 0.5
#am = 0.1
#bm = 0.1

#adj = np.array([0,1,1,0,0,0,1,1,0,0,0,1,0,0,0,0]).reshape(4,4)
#edge_list = edge_list = [(0,1),(0,2),(2,3),(1,3),(1,2)]

#a = np.array([aa, 1, aa, 1, am])
#b = np.array([1, bb, 1, bb, bm]) 
