# -*- coding: utf-8 -*-
"""
Created on 

@author: 
"""
import networkx as nx

#REVISED FUNCTION
def bearing(G,origin,destination,nodes_proj):
    import math
    node1lat = nodes_proj.at[origin, 'lat']
    node1lon = nodes_proj.at[origin, 'lon']
    node2lat = nodes_proj.at[destination, 'lat']
    node2lon = nodes_proj.at[destination, 'lon']
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

def random(G,nodes_proj,edges_proj):
    print('Random')
    
def path_length(G, path):
    
    import networkx as nx
    path_len = 0.0
    for i in range(0,len(path)-1):
        path_len = path_len + nx.shortest_path_length(G,path[i],path[i+1],weight = 'length')
    return(path_len)
    
def llf_route(G,origin,destination,nodes_proj):
    
    import networkx as nx
    
    max_depth = len(nx.dijkstra_path(G,origin,destination,weight = 'length'))
    #print('Max search depth is '+str(max_depth))
    edge_list = list(nx.bfs_predecessors(G, origin, depth_limit=max_depth))
    #node_list = list(nx.bfs_tree(G, origin, depth_limit=max_depth))
    nt_edges = list(nx.bfs_predecessors(G, origin, depth_limit=1))
    #print('Initial edges '+str(nt_edges))
    nt_nodes = list(nx.bfs_tree(G, origin, depth_limit=1))
    for i in range(len(nt_nodes)-1,len(edge_list)):
        if edge_list[i][1] in nt_nodes:
            deflection = abs(bearing(G,edge_list[i][0],edge_list[i][1],nodes_proj) - bearing(G,edge_list[i][1],origin,nodes_proj))
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

def slf_route(G,origin,destination,nodes_proj):
    route = llf_route(G,destination,origin,nodes_proj) #used opposite concept of longest leg first
    return route[::-1] #reversed

def fewest_turns(G,origin,destination,nodes_proj):
    
    import networkx as nx
    
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
                deflection = abs(bearing(G,edge_list[i][0],edge_list[i][1],nodes_proj) - bearing(G,edge_list[i][1],temp_origin,nodes_proj))
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
    return temp_route

from heapq import heappush, heappop
from itertools import count

#import networkx as nx
from networkx.utils import not_implemented_for


__all__ = ['astar_path', 'astar_path_length']


#@not_implemented_for('multigraph')
def astar_path(G, source, target, nodes_proj, heuristic=None, weight='weight'):
    if source not in G or target not in G:
        msg = 'Either source {} or target {} is not in G'
        raise nx.NodeNotFound(msg.format(source, target))

    if heuristic is None:
        # The default heuristic is h=0 - same as Dijkstra's algorithm
        def heuristic(u, v):
            return 0

    push = heappush
    pop = heappop

    c = count()
    queue = [(0, next(c), source, 0, None)]

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

                if qcost <= ncost:
                    continue

            else:
                h = heuristic(G, neighbor, target, source, nodes_proj)
            enqueued[neighbor] = ncost, h
            push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

    raise nx.NetworkXNoPath("Node %s not reachable from %s" % (source, target))



def astar_path_length(G, source, target, heuristic=None, weight='weight'):
    
    import networkx as nx
    
    if source not in G or target not in G:
        msg = 'Either source {} or target {} is not in G'
        raise nx.NodeNotFound(msg.format(source, target))

    path = astar_path(G, source, target, heuristic, weight)
    return sum(G[u][v].get(weight, 1) for u, v in zip(path[:-1], path[1:]))

def astar_least_angle(G,origin,destination,nodes_proj,weight = None,): 
    
    import networkx as nx
    
    g = nx.Graph()
    for u,v,data in G.edges(data=True):
        w = data['length'] if 'length' in data else 1.0
        if g.has_edge(u,v):
            g[u][v]['length'] += w
        else:
            g.add_edge(u, v, length=w)

    route =  astar_path(g, origin, destination, nodes_proj, heuristic=la_heuristic, weight=weight,)
    return route

def la_heuristic(G,neighbor,destination,origin,nodes_proj):
    target_angle = bearing(G,origin,destination,nodes_proj)
    least_angle = abs(target_angle - bearing(G,origin,neighbor,nodes_proj))       
    return 1000000*least_angle

def astar_least_angle_2(G,origin,destination,nodes_proj,weight = None,): 
    
    import networkx as nx
    
    g = nx.Graph()
    for u,v,data in G.edges(data=True):
        w = data['length'] if 'length' in data else 1.0
        if g.has_edge(u,v):
            g[u][v]['length'] += w
        else:
            g.add_edge(u, v, length=w)

    route =  astar_path(g, origin, destination, nodes_proj, heuristic=la_heuristic_2, weight=weight,)
    return route

def la_heuristic_2(G,neighbor,destination,origin,nodes_proj):
    target_angle = bearing(G,destination,origin,nodes_proj)
    least_angle = abs(target_angle - bearing(G,destination,neighbor,nodes_proj))       
    return 1000000*least_angle

def haversine_distance(lat1,lon1,lat2,lon2):
    
    from math import sin, cos, sqrt, atan2, radians
    
    # approximate radius of earth in km
    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c * 1000 #distance in metres
    return distance
