from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
import numpy as np
import networkx as nx

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
    

def active_linkset_animation():
    """
    Returns an animation of the active linkset with demand
    
    For this to work, the solution must also have been calculated and the demand range must be called D
    (will try to fix this...)
    """
    
    fig = plt.figure()

    graph = plot_active_links_1step(g, sols[0])

    def animate(i):
        global graph, D
        graph = plot_active_links_1step(g, sols[i])
        plt.title("demand = {}".format(D[i]))
        return graph

    ani = mpl.animation.FuncAnimation(fig, animate, frames=len(D))
    
    return ani

        
def MST(g, w='a'):
    """
    Returns the minimum spanning tree of the graph given,
    the property of the edges to use as weight is w (by default 'a')
    """
    g_mst = nx.mst.minimum_spanning_tree(g, weight=w)
    
    return g_mst

    

#plot_net(g, "edges")
#plot_net(g_mst, 'edges', 'cyan', width=2.5)
#plot_net(g_pruned, 'edges', 'magenta')