# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:14:43 2019

@author: dbhowmick
"""

import osmnx as ox
import networkx as nx
#import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import math
#import random


G = ox.graph_from_point((-37.814, 144.96332), distance=2000, distance_type='bbox', network_type='walk')
nodes, edges = ox.graph_to_gdfs(G)
fig, ax = ox.plot_graph(G)

# calculate edge bearings and visualize their frequency
G = ox.add_edge_bearings(G)
bearings = pd.Series([data['bearing'] for u, v, k, data in G.edges(keys=True, data=True)])
#lengths = pd.Series([data['length'] for u, v, k, data in G.edges(keys=True, data=True)])
ax = bearings.hist(bins=60, zorder=2, alpha=0.8)
ax.set_xlim(0, 360)
ax.set_ylim(0,60000)
ax.set_title('Melbourne pedestrian network edge bearings')
#plt.savefig('Edge_bearings_Sunbury')
plt.show()

H = ox.graph_from_point((39.9075, 116.39723), distance=35000, distance_type='bbox', network_type='walk')
nodes, edges = ox.graph_to_gdfs(H)
fig, ax = ox.plot_graph(H)

# calculate edge bearings and visualize their frequency
H = ox.add_edge_bearings(H)
bearings = pd.Series([data['bearing'] for u, v, k, data in H.edges(keys=True, data=True)])
#lengths = pd.Series([data['length'] for u, v, k, data in H.edges(keys=True, data=True)])
ax = bearings.hist(bins=60, zorder=2, alpha=0.8)
ax.set_xlim(0, 360)
ax.set_ylim(0,60000)
ax.set_title('Beijing pedestrian network edge bearings')
#plt.savefig('Edge_bearings_Sunbury')
plt.show()


G_proj = ox.project_graph(G)
fig, ax = ox.plot_graph(G_proj)
plt.tight_layout()
nodes_proj, edges_proj = ox.graph_to_gdfs(G_proj, nodes=True, edges=True)

# def bearing(G,origin,destination):
    
#     #nodes_proj, edges_proj = ox.graph_to_gdfs(G_proj, nodes=True, edges=True)
#     node1lat = nodes_proj.at[origin, 'lat']
#     node1lon = nodes_proj.at[origin, 'lon']
#     node2lat = nodes_proj.at[destination, 'lat']
#     node2lon = nodes_proj.at[destination, 'lon']
#     londiff = node2lon - node1lon 
#     latdiff = node2lat - node1lat
#     if latdiff > 0 and londiff > 0:
#         bearing = math.degrees(math.atan2(latdiff,londiff))
#     elif latdiff < 0 and londiff > 0:
#         bearing = 360.0 + math.degrees(math.atan2(latdiff,londiff))
#     elif latdiff < 0 and londiff < 0:
#         bearing = 360.0 + math.degrees(math.atan2(latdiff,londiff))
#     else:
#         bearing = math.degrees(math.atan2(latdiff,londiff))
#     return bearing
#REVISED FUNCTION
def bearing(G,node1,node2):

    node1lat = nodes.at[node1, 'y']
    node1lon = nodes.at[node1, 'x']
    node2lat = nodes.at[node2, 'y']
    node2lon = nodes.at[node2, 'x']
    londiff = node2lon - node1lon 
    print('londiff: '+str(londiff))
    latdiff = node2lat - node1lat
    print('latdiff: '+str(latdiff))
    if latdiff > 0 and londiff > 0: # Quadrant1
        bearing = 90.0 - math.degrees(math.atan2(latdiff,londiff))
    elif latdiff < 0 and londiff > 0: #Qaudrant2
        bearing = 90.0 - math.degrees(math.atan2(latdiff,londiff))
    elif latdiff < 0 and londiff < 0: #Qaudrant3
        bearing = 90.0 - math.degrees(math.atan2(latdiff,londiff))
    elif latdiff > 0 and londiff < 0: #Qaudrant4
        bearing = 450.0 - math.degrees(math.atan2(latdiff,londiff))

    return bearing

def path_length(G, path):

    path_len = 0.0
    for i in range(0,len(path)-1):
        path_len = path_len + nx.shortest_path_length(G,path[i],path[i+1],weight = 'length')
    return(path_len)
    
def llf_route(G,origin,destination):
    max_depth = len(nx.dijkstra_path(G,origin,destination,weight = 'length'))
    #print('Max search depth is '+str(max_depth))
    edge_list = list(nx.bfs_predecessors(G, origin, depth_limit=max_depth))
    #node_list = list(nx.bfs_tree(G, origin, depth_limit=max_depth))
    nt_edges = list(nx.bfs_predecessors(G, origin, depth_limit=1))
    #print('Initial edges '+str(nt_edges))
    nt_nodes = list(nx.bfs_tree(G, origin, depth_limit=1))
    for i in range(len(nt_nodes)-1,len(edge_list)):
        if edge_list[i][1] in nt_nodes:
            deflection = abs(bearing(G,edge_list[i][0],edge_list[i][1]) - bearing(G,edge_list[i][1],origin))
            if deflection < 45.0 :
                nt_nodes.append(edge_list[i][0])
                nt_edges.append(edge_list[i])
    
    if destination in nt_nodes:
        return nx.dijkstra_path(G,origin,destination,weight = 'length')
    
    last_leg = 1000000000.0
    for j in range(0,len(nt_nodes)):
        leg = nx.dijkstra_path_length(G,destination,nt_nodes[j],weight = 'length')       
        if leg < last_leg:
            last_leg = leg
            route_node = nt_nodes[j]    
    #print('Nodes are '+str(nt_nodes))
    #print('Edges are '+str(nt_edges))
    #print('Node with least distance from destination is '+str(route_node))
      
    prev_node = route_node
    route = nx.dijkstra_path(G,destination,route_node,weight = 'length')
    #print('Last route segment '+str(route))
    while prev_node != origin:
        for i in range(0,len(nt_edges)):
            if nt_edges[i][0] == prev_node:
                prev_node = nt_edges[i][1]
                route.append(prev_node)
                #print('Updated route '+str(route[::-1]))
                break
    return route[::-1] #reversing the route

#def slf_route2(G,origin,destination):
#    max_depth = len(nx.dijkstra_path(G,origin,destination,weight = 'length'))
#    #print('Max search depth is '+str(max_depth))
#    edge_list = list(nx.bfs_predecessors(G, origin, depth_limit=max_depth))
#    #node_list = list(nx.bfs_tree(G, origin, depth_limit=max_depth))
#    nt_edges = list(nx.bfs_predecessors(G, origin, depth_limit=1))
#    #print('Initial edges '+str(nt_edges))
#    nt_nodes = list(nx.bfs_tree(G, origin, depth_limit=1))
#    for i in range(len(nt_nodes)-1,len(edge_list)):
#        if edge_list[i][1] in nt_nodes:
#            deflection = abs(bearing(G,edge_list[i][0],edge_list[i][1]) - bearing(G,edge_list[i][1],origin))
#            if deflection < 45.0 :
#                nt_nodes.append(edge_list[i][0])
#                nt_edges.append(edge_list[i])
#    
#    if destination in nt_nodes:
#        return nx.dijkstra_path(G,origin,destination,weight = 'length')
#    
#    del nt_nodes[0] #removing origin from nt_nodes
#    leg_ratio = 1000000000.0
#    for j in range(0,len(nt_nodes)):
#        leg_org =  nx.dijkstra_path_length(G,origin,nt_nodes[j],weight = 'length')
#        #print('First_leg '+str(leg_org))            
#        leg_dest = nx.dijkstra_path_length(G,destination,nt_nodes[j],weight = 'length')
#        #print('Last_leg '+str(leg_dest)) 
#        ratio = 1/(leg_dest*leg_org) # lower ratio is preferred; lower first leg ensures shortest leg;#lower last leg ensures direction towards destination
#        if ratio < leg_ratio:
#            route_node = nt_nodes[j]    
#    #print('Nodes are '+str(nt_nodes))
#    #print('Edges are '+str(nt_edges))
#    #print('Node with least distance from destination is '+str(route_node))
#      
#    prev_node = route_node
#    route = nx.dijkstra_path(G,destination,route_node,weight = 'length')
#    #print('Last route segment '+str(route))
#    while prev_node != origin:
#        for i in range(0,len(nt_edges)):
#            if nt_edges[i][0] == prev_node:
#                prev_node = nt_edges[i][1]
#                route.append(prev_node)
#                #print('Updated route '+str(route[::-1]))
#                break
#    return route[::-1] #reversing the route

def slf_route(G,origin,destination):
    route = llf_route(G,destination,origin) #used opposite concept of longest leg first
    return route[::-1] #reversed


from heapq import heappush, heappop
from itertools import count

#import networkx as nx
from networkx.utils import not_implemented_for


__all__ = ['astar_path', 'astar_path_length']


@not_implemented_for('multigraph')
def astar_path(G, source, target, heuristic=None, weight='weight'):
    """Returns a list of nodes in a shortest path between source and target
    using the A* ("A-star") algorithm.

    There may be more than one shortest path.  This returns only one.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path

    heuristic : function
       A function to evaluate the estimate of the distance
       from the a node to the target.  The function takes
       two nodes arguments and must return a number.

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight.

    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> print(nx.astar_path(G, 0, 4))
    [0, 1, 2, 3, 4]
    >>> G = nx.grid_graph(dim=[3, 3])  # nodes are two-tuples (x,y)
    >>> nx.set_edge_attributes(G, {e: e[1][0]*2 for e in G.edges()}, 'cost')
    >>> def dist(a, b):
    ...    (x1, y1) = a
    ...    (x2, y2) = b
    ...    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    >>> print(nx.astar_path(G, (0, 0), (2, 2), heuristic=dist, weight='cost'))
    [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]


    See Also
    --------
    shortest_path, dijkstra_path

    """
    if source not in G or target not in G:
        msg = 'Either source {} or target {} is not in G'
        raise nx.NodeNotFound(msg.format(source, target))

    if heuristic is None:
        # The default heuristic is h=0 - same as Dijkstra's algorithm
        def heuristic(u, v):
            return 0

    push = heappush
    pop = heappop

    # The queue stores priority, node, cost to reach, and parent.
    # Uses Python heapq to keep in priority order.
    # Add a counter to the queue to prevent the underlying heap from
    # attempting to compare the nodes themselves. The hash breaks ties in the
    # priority and is guaranteed unique for all nodes in the graph.
    c = count()
    queue = [(0, next(c), source, 0, None)]

    # Maps enqueued nodes to distance of discovered paths and the
    # computed heuristics to target. We avoid computing the heuristics
    # more than once and inserting the node into the queue too many times.
    enqueued = {}
    # Maps explored nodes to parent closest to the source.
    explored = {}

    while queue:
        # Pop the smallest item from queue.
        _, __, curnode, dist, parent = pop(queue)

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path

        if curnode in explored:
            continue

        explored[curnode] = parent

        for neighbor, w in G[curnode].items():
            if neighbor in explored:
                continue
            ncost = dist + w.get(weight, 1)
            if neighbor in enqueued:
                qcost, h = enqueued[neighbor]
                # if qcost <= ncost, a less costly path from the
                # neighbor to the source was already determined.
                # Therefore, we won't attempt to push this neighbor
                # to the queue
                if qcost <= ncost:
                    continue

            else:
                h = heuristic(neighbor, target, source)
            enqueued[neighbor] = ncost, h
            push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

    raise nx.NetworkXNoPath("Node %s not reachable from %s" % (source, target))



def astar_path_length(G, source, target, heuristic=None, weight='weight'):
    """Returns the length of the shortest path between source and target using
    the A* ("A-star") algorithm.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path

    heuristic : function
       A function to evaluate the estimate of the distance
       from the a node to the target.  The function takes
       two nodes arguments and must return a number.

    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.

    See Also
    --------
    astar_path

    """
    if source not in G or target not in G:
        msg = 'Either source {} or target {} is not in G'
        raise nx.NodeNotFound(msg.format(source, target))

    path = astar_path(G, source, target, heuristic, weight)
    return sum(G[u][v].get(weight, 1) for u, v in zip(path[:-1], path[1:]))

def astar_least_turns(G,origin,destination,weight = None):   
    g = nx.Graph()
    for u,v,data in G.edges(data=True):
        w = data['length'] if 'length' in data else 1.0
        if g.has_edge(u,v):
            g[u][v]['length'] += w
        else:
            g.add_edge(u, v, length=w)

    route =  astar_path(g, origin, destination, heuristic=la_heuristic, weight=weight)
    #route_length = path_length(G,route)
    #print(route_length)
    return route

def la_heuristic(neighbor, destination, origin):
    target_angle = bearing(G,origin,destination)
    least_angle = abs(target_angle - bearing(G,origin,neighbor))       
    return 1000000*least_angle

def astar_least_angle_2(G,origin,destination,weight = None,): 
    
    import networkx as nx
    
    g = nx.Graph()
    for u,v,data in G.edges(data=True):
        w = data['length'] if 'length' in data else 1.0
        if g.has_edge(u,v):
            g[u][v]['length'] += w
        else:
            g.add_edge(u, v, length=w)

    route =  astar_path(g, origin, destination, heuristic=la_heuristic_2, weight=weight,)
    return route

def la_heuristic_2(neighbor,destination,origin):
    target_angle = bearing(G,destination,origin)
    least_angle = abs(target_angle - bearing(G,destination,neighbor))       
    return 1000000*least_angle

def astar_least_angle_3(G,origin,destination,weight = None,): 
    
    import networkx as nx
    
    g = nx.Graph()
    for u,v,data in G.edges(data=True):
        w = data['length'] if 'length' in data else 1.0
        if g.has_edge(u,v):
            g[u][v]['length'] += w
        else:
            g.add_edge(u, v, length=w)

    route =  astar_path(g, origin, destination, heuristic=la_heuristic_3, weight=weight,)
    return route

def la_heuristic_3(neighbor,destination,origin):
    target_angle = bearing(G,origin,neighbor)
    least_angle = abs(target_angle - bearing(G,neighbor,destination))       
    return 1000000*least_angle

def least_angle_route(G,origin,destination,weight = None):
    
    route = [origin]
    dead_end = [origin]
    target_angle = bearing(G,origin,destination)
    new_origin = origin
    #for k in range(0,10):
    while route[-1] != destination:
        new_origin = route[-1] # setting the last node on the developing route as the new origin
        #print('New origin is '+str(new_origin))
        target_angle = bearing(G,new_origin,destination)
        #print('Target angle is '+str(target_angle))
        list_next_nodes = [n for n in G.neighbors(new_origin)]
        #print(list_next_nodes)
        least_diff_angle = 360.0
        flag = 1
        flag3 = 0
        for i in range(0,len(list_next_nodes)):
            flag2 = 0
            for m in range(0,len(route)): # so that the a node already present in the route is not selected again
                if route[m] == list_next_nodes[i]:
                    flag2 = 1
            for j in range(0,len(dead_end)): #so that the dead end node with least angle does not get selected again
                if dead_end[j] == list_next_nodes[i]:
                    flag2 = 1
            if len(list_next_nodes) == 1 and (list_next_nodes[i] != destination and new_origin != origin): #we come back from dead end nodes if they are not the destination node
                flag = 0
                flag3 = 1
                route.remove(new_origin)
                new_origin_temp = route[-1]
                dead_end.append(new_origin)
                #print('Dead end nodes are '+str(dead_end))
            if len(route) > 1 :
                if list_next_nodes[i] != route[-2] and flag2 == 0:
                    flag3 = 1
                    angle = bearing(G,new_origin,list_next_nodes[i])
                    deflection = abs(target_angle - angle)
                    if deflection > 180.0:
                            deflection = 360.0 - deflection
                    if deflection < abs(least_diff_angle):
                        least_diff_angle = deflection
                        new_origin_temp = list_next_nodes[i] # making the chosen node as the temporary new origin after an iteration

            else:
                if list_next_nodes[i] != origin and flag2 == 0:
                    flag3 = 1
                    angle = bearing(G,new_origin,list_next_nodes[i])
                    deflection = abs(target_angle - angle)
                    if deflection > 180.0:
                            deflection = 360.0 - deflection
                    if deflection < abs(least_diff_angle):
                        least_diff_angle = deflection
                        new_origin_temp = list_next_nodes[i] # making the chosen node as the temporary new origin after an iteration

        if flag3 == 1:
            new_origin = new_origin_temp
        if flag == 1:
            if flag3 == 1:
                route.append(new_origin) #appending the next node to the route
            else:
                dead_end.append(new_origin)
                route.remove(new_origin)

    dead_end.clear()
    return route

def fewest_turns(G,origin,destination):
    final_route = []
    temp_origin = origin
    while destination not in final_route:
        #print('Temp origin is '+str(temp_origin))
        max_depth = len(nx.dijkstra_path(G,temp_origin,destination,weight = 'length'))
        #print('Max search depth is '+str(max_depth))
        edge_list = list(nx.bfs_predecessors(G, temp_origin, depth_limit=max_depth))
        #node_list = list(nx.bfs_tree(G, origin, depth_limit=max_depth))
        nt_edges = list(nx.bfs_predecessors(G, temp_origin, depth_limit=1))
        #print('Initial edges '+str(nt_edges))
        nt_nodes = list(nx.bfs_tree(G, temp_origin, depth_limit=1))
        #t_nodes = []
        for i in range(len(nt_nodes)-1,len(edge_list)):
            if edge_list[i][1] in nt_nodes:
                deflection = abs(bearing(G,edge_list[i][0],edge_list[i][1]) - bearing(G,edge_list[i][1],temp_origin))
                if deflection < 45.0 :
                    nt_nodes.append(edge_list[i][0])
                    nt_edges.append(edge_list[i])
                #else:
                    #t_nodes.append(edge_list[i][0])
        if destination in nt_nodes:
            final_route.extend(nx.dijkstra_path(G,temp_origin,destination,weight = 'length'))
            break
        
        last_leg = 1000000000.0
        for j in range(0,len(nt_nodes)):
            leg = nx.dijkstra_path_length(G,destination,nt_nodes[j],weight = 'length')       
            if leg < last_leg:
                last_leg = leg
                route_node = nt_nodes[j]
        
        
        prev_node = route_node
        route = [route_node]
        #route = nx.dijkstra_path(G,destination,route_node,weight = 'length')
        #print('Last route segment '+str(route))
        while prev_node != temp_origin:
            for i in range(0,len(nt_edges)):
                if nt_edges[i][0] == prev_node:
                    prev_node = nt_edges[i][1]
                    route.append(prev_node)
                    #print('Updated route '+str(route[::-1]))
                    break
        
        temp_origin =  route_node
        final_route.extend(route[::-1])
        #print('Final updated rouet '+str(final_route))
    #return final_route
    temp_route = []
    temp_route.extend(final_route)
    flag = 0
    for i in range(0,len(final_route)-1):
        for j in range(i+1,len(final_route)):
            if final_route[i] == final_route[j]:
#                print('i is '+str(i))
#                print('Node is '+str(route[i]))
#                print('j is '+str(j))
#                print('Node is '+str(route[j]))
#                print('Removed portion '+str(temp_route[i:j]))
                del temp_route[i:j]
                flag = 1
                break
        if flag == 1:
             break
    from itertools import groupby
    route = [x[0] for x in groupby(route)]
    return temp_route

# degrees to radians
def deg2rad(degrees):
    return math.pi*degrees/180.0
# radians to degrees
def rad2deg(radians):
    return 180.0*radians/math.pi

# Semi-axes of WGS-84 geoidal reference
WGS84_a = 6378137.0  # Major semiaxis [m]
WGS84_b = 6356752.3  # Minor semiaxis [m]

# Earth radius at a given latitude, according to the WGS-84 ellipsoid [m]
def WGS84EarthRadius(lat):
    # http://en.wikipedia.org/wiki/Earth_radius
    An = WGS84_a*WGS84_a * math.cos(lat)
    Bn = WGS84_b*WGS84_b * math.sin(lat)
    Ad = WGS84_a * math.cos(lat)
    Bd = WGS84_b * math.sin(lat)
    return math.sqrt( (An*An + Bn*Bn)/(Ad*Ad + Bd*Bd) )

# Bounding box surrounding the point at given coordinates,
# assuming local approximation of Earth surface as a sphere
# of radius given by WGS84
def boundingBox(latitudeInDegrees, longitudeInDegrees, halfSideInKm):
    lat = deg2rad(latitudeInDegrees)
    lon = deg2rad(longitudeInDegrees)
    halfSide = 1000*halfSideInKm

    # Radius of Earth at given latitude
    radius = WGS84EarthRadius(lat)
    # Radius of the parallel at given latitude
    pradius = radius*math.cos(lat)
    
    
    latMin = lat - halfSide/radius
    latMax = lat + halfSide/radius
    lonMin = lon - halfSide/pradius
    lonMax = lon + halfSide/pradius

    bbox = [rad2deg(latMin),rad2deg(latMax),rad2deg(lonMin),rad2deg(lonMax)]
    return bbox

import pandas as pd
#import math
import random

m = 0
list_origin = []
list_destination = []
list_shortest = []
#list_kpath = []
#list_leastangle = []
list_leastangle_astar = []
list_leastangle_2 = []
list_leastangle_3 = []
list_llf = []
list_slf = []
list_fturns = []
list_shortest_route = []
#list_kpath = []
#list_leastangle = []
list_leastangle_astar_route = []
list_leastangle_2_route = []
list_leastangle_3_route = []
list_llf_route = []
list_slf_route = []
list_fturns_route = []
#list_mturns = []
df = pd.DataFrame()
while m <50:
    print('Iteration number '+str(m+1))

    origin = random.choice(list(nodes_proj['osmid']))
    destination  = random.choice(list(nodes_proj['osmid']))
    print('Origin is '+str(origin))
    print('Destination is '+str(destination))
    #avoiding same OD pair
    if m>1:
        for n in range(1,m):
            if list_origin[n] == origin and list_destination[n] == destination:
                continue
    
    #clearing buffer
    if boundingBox(-37.814, 144.96332, 2.0)[0] <= nodes_proj.at[origin, 'lat'] <= boundingBox(-37.814, 144.96332, 2.0)[1]:
        if boundingBox(-37.814, 144.96332, 2.0)[2] <= nodes_proj.at[origin, 'lon'] <= boundingBox(-37.814, 144.96332, 2.0)[3]:
            #checking for min and max trip length thresholds
            shortest_route_length = path_length(G_proj, nx.shortest_path(G_proj, origin,destination,weight = 'length'))
            if shortest_route_length >= 400 and shortest_route_length <= 2000 :
                m = m+1 #counting a valid walking trip
        #        
                #list_paths = k_shortest_paths(G_proj, origin,destination, 1000, weight = 'length')
        #        
                list_origin.append(origin)
                list_destination.append(destination)
                
                list_shortest.append(shortest_route_length)
                #list_kpath.append(path_length(G_proj,list_paths[-1]))
                #list_leastangle.append(path_length(G_proj,least_angle_route(G_proj,origin,destination)))
                list_leastangle_astar.append(path_length(G_proj,astar_least_turns(G_proj,origin,destination)))       
                list_leastangle_2.append(path_length(G_proj,astar_least_angle_2(G_proj,origin,destination)))
                list_leastangle_3.append(path_length(G_proj,astar_least_angle_3(G_proj,origin,destination)))
                list_llf.append(path_length(G_proj,llf_route(G_proj,origin,destination)))
                list_slf.append(path_length(G_proj,slf_route(G,origin,destination)))     
                list_fturns.append(path_length(G_proj,fewest_turns(G_proj,origin,destination)))
                
                list_shortest_route.append(nx.shortest_path(G_proj, origin,destination,weight = 'length'))
                list_leastangle_astar_route.append(astar_least_turns(G_proj,origin,destination))     
                list_leastangle_2_route.append(astar_least_angle_2(G_proj,origin,destination))
                list_leastangle_3_route.append(astar_least_angle_3(G_proj,origin,destination))
                list_llf_route.append(llf_route(G_proj,origin,destination))
                list_slf_route.append(slf_route(G,origin,destination))
                list_fturns_route.append(fewest_turns(G_proj,origin,destination))
                #list_mturns.append(path_length(G_proj,most_turns_route(G_proj,origin,destination,list_paths,45)))
    
#        
#
df['Origin_node'] = list_origin
df['Destination_node'] = list_destination
df['Shortest_route_length'] = list_shortest
#df['K-th_route_length'] = list_kpath
#df['Least_angle_route_length'] = list_leastangle
df['Least_angle_astar'] = list_leastangle_astar
df['Least_angle_astar_2'] = list_leastangle_2
df['Least_angle_astar_3'] = list_leastangle_3
df['Longest_leg_first_route_length'] = list_llf
df['Shortest_leg_first_route_length'] = list_slf
df['Fewest_turns_route_length'] = list_fturns
df['Shortest_route'] = list_shortest
#df['K-th_route_length'] = list_kpath
#df['Least_angle_route_length'] = list_leastangle
df['Least_angle_astar_route'] = list_leastangle_astar_route
df['Least_angle_astar_2_route'] = list_leastangle_2_route
df['Least_angle_astar_3_route'] = list_leastangle_3_route
df['Longest_leg_first_route'] = list_llf_route
df['Shortest_leg_first_route'] = list_slf_route
df['Fewest_turns_route'] = list_fturns_route
#df['Most_turns_route_length'] = list_mturns
#
df.to_csv('Melbourne_s50000_d2000_buffer.csv')

fig, ax = ox.plot_graph_route(G, df['Least_angle_astar_route'][45], fig_height=20)
fig, ax = ox.plot_graph_route(G, df['Least_angle_astar_2_route'][45], fig_height=20)
fig, ax = ox.plot_graph_route(G, df['Longest_leg_first_route'][45], fig_height=20)
fig, ax = ox.plot_graph_route(G, df['Shortest_leg_first_route'][45], fig_height=20)
fig, ax = ox.plot_graph_route(G, df['Fewest_turns_route'][45], fig_height=20)
from itertools import groupby
fig, ax = ox.plot_graph_route(G, [x[0] for x in groupby(df['Fewest_turns_route'][36])], fig_height=20)
plt.show()

fig, ax = ox.plot_graph_route(G, [6167441044,6167289287], fig_height=10)

df['Least_angle_astar'].mean()
df['Least_angle_astar'].std()
df['Least_angle_astar_2'].mean()
df['Least_angle_astar_2'].std()

from scipy.stats import ttest_ind
ttest_ind(df['Least_angle_astar'], df['Least_angle_astar_2'])
ttest_ind(df['Least_angle_astar'], df['Least_angle_astar_3'])
ttest_ind(df['Least_angle_astar'], df['Longest_leg_first_route_length'])
ttest_ind(df['Least_angle_astar'], df['Shortest_leg_first_route_length'])
ttest_ind(df['Longest_leg_first_route_length'], df['Shortest_leg_first_route_length'])
