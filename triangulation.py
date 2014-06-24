from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import scipy as sp
import numpy as np
import networkx as nx


mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14


"""
Functions and/or objects to create a network using delaunay triangulation.
The objective is to generate a random connected directed planar graph.
"""

def make_triangulation(points, b_max=1.0):
    """
    Makes a networkx graph from a delaunay triangulation in a unit square
    
    points - number of nodes for the graph
    """
    
    number_of_points = points
    pts = [np.random.random(2) for i in range(number_of_points)]
    
    pts = np.array(pts)

    tri = Delaunay(pts)

    edges = set()
    edge_lengths = set()
    for n in xrange(tri.nsimplex):
        
        vertices = tri.vertices[n]
        
        edge_length = np.linalg.norm(tri.points[vertices[0]] - tri.points[vertices[1]])
        edge = (vertices[0], vertices[1], edge_length)
        edges.add(edge)
        edge_lengths.add(edge_length)
        
        edge_length = np.linalg.norm(tri.points[vertices[0]] - tri.points[vertices[2]])
        edge = (vertices[0], vertices[2], edge_length)
        edges.add(edge)
        edge_lengths.add(edge_length)
        
        edge_length = np.linalg.norm(tri.points[vertices[1]] - tri.points[vertices[2]])
        edge = (vertices[1], vertices[2], edge_length)
        edges.add(edge)
        edge_lengths.add(edge_length)

        
    point_dict = dict(zip(range(len(pts)), pts))
    position_dicts = [{"pos": i} for i in pts]
    point_list = [(i, position_dicts[i]) for i in range(number_of_points)]
    edge_list = list(edges)
    edge_lengths = list(edge_lengths)

    g = nx.Graph()
    g.add_nodes_from(point_list)
    g.add_weighted_edges_from(edge_list, weight='a')

    a_list = edge_lengths
    b_list = [b_max*np.random.random() for i in range(len(g.edges()))] #uniform at the moment but can be different (uniform in the range 0-b_range)

    for i in range(len(edge_list)):
            g.edge[edge_list[i][0]][edge_list[i][1]]['b'] = b_list[i]


    norms = [np.linalg.norm(g.node[i]['pos']) for i in g.node]

    origin = np.argmin(norms)
    destination = np.argmax(norms)

    g.node[origin]['origin']=True
    g.node[destination]['destination']=True
    
    return g
    
    
    
    
def prune_triangulation(g, param):
    """
    Removes a proportion of links of graph g according to param
    
    if param = 1 all links are removed
    """
    
    g_removing = g.copy()

    for i in range(len(g.nodes())):
        if g.node[i].has_key('origin') and g.node[i]['origin']:
            origin = i
        if g.node[i].has_key('destination') and g.node[i]['destination']:
            destination = i
    
    total_edges = len(g.edges())
    to_remove = int(total_edges*param)

    eds = list(g.edges(data=True))

    while to_remove > 0:
        
        if len(eds) != 0:    
            
            np.random.shuffle(eds)
            
            e = eds.pop()
            
            g_removing.remove_edge(*e[:2])
            
            if nx.has_path(g_removing, origin, destination):
                to_remove -= 1
            else:
                
                g_removing.add_edge(*e)
            
        else:
            print "cannot remove more edges and have O-D connected"
            break
        
    #extract subgraph from connected component of OD pair
    connected_nodes =  nx.node_connected_component(g_removing, origin)
    g_pruned = nx.subgraph(g_removing, connected_nodes)


    #Now we rename the nodes to avoid having skipped numbers
    g_pruned = nx.convert_node_labels_to_integers(g_pruned,first_label=0)
    g_pruned = g_pruned.to_directed()
    
    return g_pruned


def make_pruned_triangulation(nodes, param=0, b_max=1.0):
    g = make_triangulation(nodes, b_max)
    g = prune_triangulation(g, param)
    
    return g

    
def reasign_a(g, a_max):
    """
    Reasingns the 'a' parameter on the edges of graph 'g'

    g - network
    """

    for e in g.edges():
        g[e[0]][e[1]]['a'] = np.random.random()*a_max


def rand_shrink_a(g, percentage_of_links, scale):
    """
    This function reasigns the "a" parameter of a percentage of links in the network to make them small
    it scales them by the scaling factor scale
    """

    edge_list = g.edges()
    number_to_change = int(len(edge_list)*percentage_of_links)
    
    np.random.shuffle(edge_list)
    edges_to_change = edge_list[:number_to_change]

    for e in edges_to_change:
        g[e[0]][e[1]]["a"] = g[e[0]][e[1]]["a"]*scale
    

def reasign_b(g, b_max):
    """
    Reasingns the 'b' parameter on the edges of graph 'g'
    
    g - network
    """
     
    for e in g.edges():
        g[e[0]][e[1]]['b'] = np.random.random()*b_max
     
    

    
    
    
    
    
    
    
#########################################################
#########################################################
#########################################################
    
    

#number_of_points = 50
#param = 0.5
##min_allowed_dist = 0.025
#pts = [np.random.random(2) for i in range(number_of_points)]

##removed = True
##counter = 0
##while removed:
    
    ##to_remove = []
    ##for i in range(len(pts)):
        ##for j in range(i+1, len(pts)):
            
            ##if np.linalg.norm(pts[i] - pts[j]) < min_allowed_dist:
                ##to_remove.append(j)
    
    ##if len(to_remove) != 0:
        ##new_pts = [pts[i] for i in range(len(pts)) if  not(i in to_remove)]
        
        ##for i in range(len(to_remove)):
            ##new_point = np.random.random(2)
            
            ##dists = [np.linalg.norm(new_point - p) for p in new_pts]
    ##else:
        ##removed = False
            
    ##counter +=1
            
#pts = np.array(pts)
            
            

#tri = Delaunay(pts)

#edges = set()
#edge_lengths = set()
#for n in xrange(tri.nsimplex):
    
    #vertices = tri.vertices[n]
    
    #edge_length = np.linalg.norm(tri.points[vertices[0]] - tri.points[vertices[1]])
    #edge = (vertices[0], vertices[1], edge_length)
    #edges.add(edge)
    #edge_lengths.add(edge_length)
    
    #edge_length = np.linalg.norm(tri.points[vertices[0]] - tri.points[vertices[2]])
    #edge = (vertices[0], vertices[2], edge_length)
    #edges.add(edge)
    #edge_lengths.add(edge_length)
    
    #edge_length = np.linalg.norm(tri.points[vertices[1]] - tri.points[vertices[2]])
    #edge = (vertices[1], vertices[2], edge_length)
    #edges.add(edge)
    #edge_lengths.add(edge_length)

    
#point_dict = dict(zip(range(len(pts)), pts))
#position_dicts = [{"pos": i} for i in pts]
#point_list = [(i, position_dicts[i]) for i in range(number_of_points)]
#edge_list = list(edges)
#edge_lengths = list(edge_lengths)

#g = nx.Graph()
#g.add_nodes_from(point_list)
#g.add_weighted_edges_from(edge_list, weight='a')

#a_list = edge_lengths
#b_list = [np.random.random() for i in range(len(g.edges()))] #uniform at the moment but can be different

#for i in range(len(edge_list)):
        #g.edge[edge_list[i][0]][edge_list[i][1]]['b'] = b_list[i]


#norms = [np.linalg.norm(g.node[i]['pos']) for i in g.node]

#origin = np.argmin(norms)
#destination = np.argmax(norms)

#g.node[origin]['origin']=True
#g.node[destination]['destination']=True

##Remove edges from graph

#g_removing = g.copy()

#total_edges = len(g.edges())
#to_remove = int(total_edges*param)


#eds = list(g.edges(data=True))

#while to_remove > 0:
    
    #if len(eds) != 0:    
        
        #np.random.shuffle(eds)
        
        #e = eds.pop()
        
        #g_removing.remove_edge(*e[:2])
        
        #if nx.has_path(g_removing, origin, destination):
            #to_remove -= 1
        #else:
            
            #g_removing.add_edge(*e)
        
    #else:
        #print "cannot remove more edges and have O-D connected"
        #break
    
##extract subgraph from connected component of OD pair
#connected_nodes =  nx.node_connected_component(g_removing, origin)
#g_pruned = nx.subgraph(g_removing, connected_nodes)


##Now we rename the nodes to avoid having skipped numbers
#g_pruned = nx.convert_node_labels_to_integers(g_pruned,first_label=0)
#g_pruned = g_pruned.to_directed()




##nx.draw_networkx_nodes(g, point_dict, node_size=20, linewidths=0.5 )
##nx.draw_networkx_edges(g, point_dict)


##fig = plt.gcf()
##plt.axis('off')
##plt.axis('equal')

###fig.savefig('out.png', bbox_inches='tight', pad_inches=0)

##plt.show() 



def plot_net(g, draw='all', edge_colour='k', width=1.0):
    
    
    for i in range(len(g.nodes())):
        if g.node[i].has_key('origin') and g.node[i]['origin']:
            origin = i
        if g.node[i].has_key('destination') and g.node[i]['destination']:
            destination = i
    
    point_dict = {i:g.node[i]['pos'] for i in g.nodes()}
    
    if draw == 'all':
        nx.draw_networkx_nodes(g, point_dict, node_size=20, linewidths=0.5 )
        nx.draw_networkx_edges(g, point_dict, edge_color=edge_colour, )
        
        fig = plt.gcf()
        plt.axis('off')
        plt.axis('equal')
        
    elif draw == 'nodes':
        nx.draw_networkx_nodes(g, point_dict, node_size=20, linewidths=0.5 )
        
        fig = plt.gcf()
        plt.axis('off')
        plt.axis('equal')
        
    elif draw == 'edges':
        nx.draw_networkx_edges(g, point_dict, edge_color=edge_colour, width=width)
        
        fig = plt.gcf()
        plt.axis('equal')
        plt.axis('off')
        
    nx.draw_networkx_nodes(g, point_dict, nodelist=[origin], node_color='g', node_size=30)
    nx.draw_networkx_nodes(g, point_dict, nodelist=[destination], node_color='b', node_size=30)
        
    plt.show()
    
    
def plot_active_links_1step(g, sol, colour='magenta'):
    """
    Plots the active link set of the graph.
    
    Active links in desired colour (default magenta), inactive links in light gray
    """
    
    for i in range(len(g.nodes())):
        if g.node[i].has_key('origin') and g.node[i]['origin']:
            origin = i
        if g.node[i].has_key('destination') and g.node[i]['destination']:
            destination = i
    
    point_dict = {i:g.node[i]['pos'] for i in g.nodes()}
    
    
    status_list = []
    tolerance = 1E-3 
    for flow in sol:
        if flow > tolerance: 
            status = 'active'
        else:
            status = 'inactive'
        
        status_list.append(status)
        
    active_edges = [g.edges()[i] for i in range(len(status_list)) if status_list[i]=='active']
    inactive_edges = [g.edges()[i] for i in range(len(status_list)) if status_list[i]=='inactive']
    
    plt.clf()
    
    nx.draw_networkx_nodes(g, point_dict, node_size=20, linewidths=0.5)
    
    nx.draw_networkx_edges(g, point_dict, edgelist=inactive_edges, edge_color='0.5', edge_cmap = mpl.cm.binary, edge_vmin=0.0, edge_vmax=1.0)
    nx.draw_networkx_edges(g, point_dict, edgelist=active_edges, edge_color=colour)
    
    nx.draw_networkx_nodes(g, point_dict, node_size=20, linewidths=0.5 )
    
    nx.draw_networkx_nodes(g, point_dict, nodelist=[origin], node_color='g', node_size=30)
    nx.draw_networkx_nodes(g, point_dict, nodelist=[destination], node_color='b', node_size=30)
    
    fig = plt.gcf()
    plt.axis('off')
    plt.axis('equal')
    
    plt.show()
    

def active_linkset_animation(g, sols):
    """
    Returns an animation of the active linkset with demand
    
    For this to work, the solution must also have been calculated and the demand range must be called D
    (will try to fix this...)
    """

    global graph
    
    fig = plt.figure()

    graph = plot_active_links_1step(g, sols[0])

    def animate(i):
        global graph, D
        graph = plot_active_links_1step(g, sols[i])
        plt.title("demand = {}".format(D[i]))
        return graph

    ani = animation.FuncAnimation(fig, animate, frames=len(D))
    
    return ani


def switch_counter(X):
    """
    counts how many times there is a change from a series of zeros to a series of ones
    in an array or list of only zeros or ones

    returns a list that has the number of swithces in a link plus the demand at which the swithces occur.
    """

    counter = 0
    switching_demand = []
    
    for i in range(1,len(X)-2):

        dif = X[i+1] - X[i]
        if dif != 0:
            counter +=1
            switching_demand.append(i)
            
    return [counter, switching_demand]

def link_state(D, flow, tol=1E-4):
    """
    Returns an array of 0s and 1s that represent whether the
    link is active for the corresponding demand levels

    flow - array that contains the flow on the link
    tol - tolerance for when considering a link is active
    """

    flow = flow/D
    
    link_status_list = []

    for f in flow:
        if f >= tol:
            status = 1 # link is active
        else:
            status = 0 # link is inactive

        link_status_list.append(status)

    return np.array(link_status_list)


    
def network_state(D, sols, tol=1E-4):
    """
    Returns a 2D array of the state of the links (active/inactive) at the demand levels (D and sols must be compatible)
    
    sols - solutions to static TA
    tol - tolerance to determine when a link has no flow
    """

    state_list = [link_state(D, x, tol) for x in sols.T]

    return np.array(state_list)


def plot_swithces(g, D, sols, tol=1E-3, col_map='RdBu'):

    states = network_state(D, sols, tol)

    y = np.arange(g.number_of_edges()+1)
    x = list(D)
    x.append(D[-1] + (D[-1] - D[-2]))
    X, Y = np.meshgrid(x, y)
    
    colormap = plt.cm.get_cmap(col_map, 2)
    plt.pcolormesh(X, Y, states, cmap=colormap)
    
    plt.hlines(np.arange(0,g.number_of_edges()), 0, D[-1])
    plt.yticks(np.arange(0,g.number_of_edges())+0.5, np.arange(1,g.number_of_edges()+1))
    cbar = plt.colorbar(ticks=[0.25, 0.75])
    cbar.set_ticklabels(['Off', 'On'], update_ticks=True)

    plt.xlabel("Demand", fontsize=16)
    plt.ylabel("Links", fontsize=16)
    plt.xlim([D[0], D[-1]])
    plt.ylim([0,g.number_of_edges()])

    plt.show()
    

#def has_switching_off(D, sols, tol=1E-4):
    #"""
    #Determines whether there are any switches in the active link set for the given demand range

    #D - demand range
    #sols - solutions for flows in the TA problem
    #tol - tolerance to determine whether there is actually a flow
    #"""

    #states = network_state(D, sols, tol)

    #counter_demand_list = [switch_counter(s) for s in states]

    #links_with_off = [i for i in range(len(sols.T)) if counter_demand_list[i][0] > 1]

    #links_with_multiple = [i for i in range(len(sols.T)) if counter_demand_list[i][0] > 2]


    #if len(links_with_off) > 0:
        #print "The network has {} links that switch off\n".format(len(links_with_off))
        #for i in range(len(links_with_off)):
            #print "Link {} switches off at {}".format(links_with_off[i], counter_demand_list[links_with_off[i]][1])

            #if len(links_with_multiple) > 0:
                #print "The network has {} links that switch off\n".format(len(links_with_multiple))
                #for i in range(len(links_with_multiple)):
                    #print "Link {} has swithes at: \t".format(links_with_off[i]),  counter_demand_list[links_with_off[i]][:]

                    #return counter_demand_list, links_with_off, links_with_multiple


def has_switching_off(D, sols, tol=1E-3):
    """
    Determines whether there are any switches in the active link set for the given demand range

    D - demand range
    sols - solutions for flows in the TA problem
    tol - tolerance to determine whether there is actually a flow
    """

    states = network_state(D, sols, tol)

    counter_demand_list = [switch_counter(s) for s in states]

    #links_with_off = [i for i in range(len(sols.T)) if counter_demand_list[i][0] > 1]

    #links_with_multiple = [i for i in range(len(sols.T)) if counter_demand_list[i][0] > 2]


    switch_types = []
    
    for i in range(len(counter_demand_list)):
        if counter_demand_list[i][0]!=0:
            for j in counter_demand_list[i][1]:
                switch_types.append(states[i,j+1])

                
    #print switch_types
    
    if 0 in switch_types:
        return True
    else:
        return False

    
def MST(g, w='a'):
    """
    Returns the minimum spanning tree of the graph given,
    the property of the edges to use as weight is w (by default 'a')
    """
    g_mst = nx.mst.minimum_spanning_tree(g, weight=w)
    
    return g_mst






###################################
###################################

# other useful functions!

def make_asym_braess(alpha, beta, xi, eta, a=0.5, b=0.5, am=0.1, bm=0.1):

    adj = np.array([0,1,1,0,
                    0,0,1,1,
                    0,0,0,1,
                    0,0,0,0]).reshape(4,4)

    a = 0.5
    b = 0.5
    am = 0.1
    bm = 0.1

    a_coefs = [a + alpha/2.0, 1 + xi/2.0 , am, 1 - xi/2.0, a - alpha/2.0]
    b_coefs = [1 + beta/2.0 , b + eta/2.0, bm, b - eta/2.0, 1 - beta/2.0]

    if all(a_coefs) and all(b_coefs):
        return make_network(adj, a_coefs, b_coefs)
    else:
        print "ERROR: Negative cost coefficient present"


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














    
#plot_net(g, "edges")
#plot_net(g_mst, 'edges', 'cyan', width=2.5)
#plot_net(g_pruned, 'edges', 'magenta')