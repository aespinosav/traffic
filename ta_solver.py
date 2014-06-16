import sys
import os
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import networkx as nx

def ta_solve(adj, edge_list, a_coefs, b_coefs, d, regime="SO", origin_index=0, destination_index=-1):
    """
    Solves the traffic assignment problem for 1 O-D pair.
    for linear cost functions of the type f = a + bx.
    
    Solves for either UE or SO passed as a string as the regime argument

    adj - adjacency matrix of traffic network
    edge_list - ordered edge list (edges represented by tuples of nodes eg. (0,1) )
    a - array of "a" parameters of link cost functions (order according to edge_list)
    b - array of "b" parameters of link cost functions (order according to edge_list)
    
    d - demand on the network
    
    By default node 0 should be the origin
    the last node should be the destination
    
    The index of the origin and destination nodes can otherwise be 
    specified in the respective arguments
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
    b[origin_index] = -d
    b[destination_index] = d
    
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
    
def ta_range_solve(D, adj, edge_list, a_coefs, b_coefs, regime="SO", origin_index=0, destination_index=-1,  fname="solver_out.txt"):
    """
    Solves the traffic assignment for a range of demand using ta_solve.
    """
    
    # We want to redirect the printed output of the solver to 
    # a file so that it does not clog the terminal or notebook or whatever
    os.system('rm {}'.format(fname));
    orig_stdout = sys.stdout
    f = file('{}'.format(fname), 'a')
    sys.stdout = f
    
    sols = []
    for d in D:
        x = ta_solve(adj, edge_list, a_coefs, b_coefs, d, regime, origin_index, destination_index)
        sols.append(x)
        
     #after the solver sends its output to file we want to restablish stdout
    sys.stdout = orig_stdout
    f.close()
    
    return np.array(sols)
    
    

def ta_solve_network(g, demand, regime, fname="solver_out.txt"):
    """Solves the traffic assignment problem for a network.
    
    g - networkx network object
    demand - a demand range for which to solve the ta problem
    """
    
    adj = np.array(nx.adjacency_matrix(g))
    edge_list = g.edges()
    
    coefs = np.array([(g[e[0]][e[1]]['a'], g[e[0]][e[1]]['b'])  for e in g.edges()])
    
    a_coefs = list(coefs[:,0])
    b_coefs = list(coefs[:,1])


    # Start by assuming no origin and destination have been assigned
    origin_exists = False
    destination_exists = False

    while (not origin_exists) and (not destination_exists):

        for i in range(len(g.nodes())):
            if g.node[i].has_key('origin'):
                origin_index = i
                origin_exists = True
            if g.node[i].has_key('destination'):
                destination_index = i
                destination_exists = True

        # If no origin or destination exists, the first node in the network is taken to be origin and
        # the last node is taken to be the destination
        if not origin_exists:
            g.node[0]['origin']=True
        if not destination_exists:
            g.node[g.number_of_nodes() - 1]['destination']=True
    
    sols = ta_range_solve(demand, adj, edge_list, a_coefs, b_coefs, regime, origin_index, destination_index, fname=fname)
    
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

    
def total_cost(x, a, b):
    """
    Calculates the total cost for a given demand on the network
    with linear cost functions.
    
    x - vector of link flows at a given demand
    a - vector of constant terms in cost functions
    b - vector of linear coefficients of cost funtions
    
    cost functions allowed are of the form f(x) = a + bx.
    """
    
    x = np.array(x)
    a = np.array(a)
    b = np.array(b)
    
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
    
def total_network_cost(g, flows):
    
    a = [g.edges(data=True)[i][-1]['a'] for i in range(g.number_of_edges())]
    b = [g.edges(data=True)[i][-1]['b'] for i in range(g.number_of_edges())]
    
    
    return total_cost_func(flows, a, b)
    
def plot_flows(D, sols, normalised=False, legend=False):
    """
    Plots the flows obtained from ta_range_solve when given the demand range and the flow solutions
    """
    if not normalised:
        for i in range(len(sols[0])):
            plt.plot(D, sols[:,i], label="Flow {}".format(i+1))
    else:
        for i in range(len(sols[0])):
            plt.plot(D, sols[:,i]/D, label="Flow {}".format(i+1))
    
    if legend:
        plt.legend(loc=4)
        
    plt.xlabel('Demand')
    plt.ylabel('Flow')
    
    #todo: plot the graph here as well (possibly in upper left corner) / maybe if graph is smaller than N nodes
        
    plt.show()

def plot_cost(g, D, sols):

    cost = total_network_cost(g, sols)

    plt.plot(D, cost)
    show()

    
    
    
def subgraph_cost(g, flows, links):
    """
    Calculates the costs as a function of demand for the flow on the links selected

    g - graph
    D - Demand range
    sols - flow solution vector for complete network
    links - either one link or a list of links to calculate the aggregate cost for
    """

    links = list(links)

    a = [g.edges(data=True)[i][-1]['a'] for i in range(g.number_of_edges())]
    b = [g.edges(data=True)[i][-1]['b'] for i in range(g.number_of_edges())]

    flow_list = np.array([flows.T[i-1] for i in links]).T

    new_a = [a[i-1] for i in links]
    new_b = [b[i-1] for i in links]

    sub_graph_costs = total_cost_func(flow_list, new_a, new_b)

    return sub_graph_costs

#def subgraph_between_nodes(start_node, finish_node):
    """
    Returns a subgraph contained between chosen nodes.
    i.e. this subgraph is the graph induced by all the possible paths from start_node
    to finish_node
    """

    
    
    



    
##Testing values (this should be commented out when using as a library)
#aa = 0.5
#bb = 0.5
#am = 0.1
#bm = 0.1

#adj = np.array([0,1,1,0,0,0,1,1,0,0,0,1,0,0,0,0]).reshape(4,4)
#edge_list = edge_list = [(0,1),(0,2),(2,3),(1,3),(1,2)]

#a = np.array([aa, 1, aa, 1, am])
#b = np.array([1, bb, 1, bb, bm]) 
