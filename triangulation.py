from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import networkx as nx

"""
Functions and/or objects to create a network using delaunay triangulation.
The objective is to generate a random connected directed planar graph.
"""


number_of_points = 1000
param = 0.95
#min_allowed_dist = 0.025
pts = [np.random.random(2) for i in range(number_of_points)]

#removed = True
#counter = 0
#while removed:
    
    #to_remove = []
    #for i in range(len(pts)):
        #for j in range(i+1, len(pts)):
            
            #if np.linalg.norm(pts[i] - pts[j]) < min_allowed_dist:
                #to_remove.append(j)
    
    #if len(to_remove) != 0:
        #new_pts = [pts[i] for i in range(len(pts)) if  not(i in to_remove)]
        
        #for i in range(len(to_remove)):
            #new_point = np.random.random(2)
            
            #dists = [np.linalg.norm(new_point - p) for p in new_pts]
    #else:
        #removed = False
            
    #counter +=1
            
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
b_list = [np.random.random() for i in range(len(g.edges()))] #uniform at the moment but can be different

for i in range(len(edge_list)):
        g.edge[edge_list[i][0]][edge_list[i][1]]['b'] = b_list[i]


norms = [np.linalg.norm(g.node[i]['pos']) for i in g.node]

origin = np.argmin(norms)
destination = np.argmax(norms)

#Remove edges from graph

g_removing = g.copy()

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



#nx.draw_networkx_nodes(g, point_dict, node_size=20, linewidths=0.5 )
#nx.draw_networkx_edges(g, point_dict)


#fig = plt.gcf()
#plt.axis('off')
#plt.axis('equal')

##fig.savefig('out.png', bbox_inches='tight', pad_inches=0)

#plt.show() 



def plot_net(g, draw='all', edge_colour='k', width=1.0):
    
    #point_dict = 
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
        

g_mst = nx.mst.minimum_spanning_tree(g, weight='a')

plot_net(g, "edges")
plot_net(g_mst, 'edges', 'cyan', width=2.5)
plot_net(g_pruned, 'edges', 'magenta')


#plt.show()
#def rand_triangulation(number_of_points):
    #"""
    #Makes a random network from a Delaunay triangulation.
    #The nodes are random points in the unit square
    
    #Takes number of points wanted in the network and 
    #returns a networkx graph object with weighted edges.
    
    #The weight of the edges is the length of the edge
    #"""
    
    #pts = [np.random.random(2) for i in range(number_of_points)]
    #pts = np.array(pts)

    #tri = Delaunay(pts)

    #edges = set()
    #for n in xrange(tri.nsimplex):
        
        #vertices = tri.vertices[n]
        
        #edge = (vertices[0], vertices[1])
        #edges.add(edge)
        
        #edge = (vertices[0], vertices[2])
        #edges.add(edge)
        
        #edge = (vertices[1], vertices[2])
        #edges.add(edge)

    #g = nx.Graph(list(edges))
    
    

            