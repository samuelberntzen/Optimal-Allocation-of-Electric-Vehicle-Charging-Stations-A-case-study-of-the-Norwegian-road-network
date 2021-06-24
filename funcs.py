import networkx as nx
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import numpy as np
import scipy.special
import shapely
from shapely.geometry import Point, Polygon, LineString 
from shapely.ops import nearest_points
from scipy import sparse
import grinpy
import time

def build_network_data(GeoData):    
    """
    Returns node-dict and edge-dict which is networkx compatible network from data available at https://kartkatalog.geonorge.no/metadata/statens-vegvesen/nvdb-ruteplan-nettverksdatasett/8d0f9066-34f9-4423-be12-8e8523089313.

    Returns node and edge dictionary.
    """

    # Get unique nodes
    nodes = {}

    print("Beginning tonodes...")
    for i in GeoData.drop_duplicates('tonode').index:

        # Get node ID for filtering duplicates
        _id          = GeoData.iloc[i]['tonode']

        # Only keep unique observations
        if _id not in nodes:
            # Get attributes if applicable, else get centroid coordinate in linestring (error usually indicate roundabout as one edge)
            try:
                x        = GeoData.iloc[i]['geometry'].boundary[-1].x
                y        = GeoData.iloc[i]['geometry'].boundary[-1].y
                roadclass = int(GeoData.iloc[i]['funcroadclass'])
                isBridge = int(GeoData.iloc[i]['isbridge'])
                isTunnel = int(GeoData.iloc[i]['istunnel'])
                geometry = Point(x,y)
            except Exception as e:
                x        = GeoData.iloc[i]['geometry'].centroid.x
                y        = GeoData.iloc[i]['geometry'].centroid.y
                roadclass = int(GeoData.iloc[i]['funcroadclass'])
                isBridge = int(GeoData.iloc[i]['isbridge'])
                isTunnel = int(GeoData.iloc[i]['istunnel'])
                geometry = Point(x,y)

            # Save and append
            content = {'x':x,'y':y,'osmid':_id, 'roadclass': roadclass, 'isBridge': isBridge, 'isTunnel': isTunnel, 'geometry':geometry}
            nodes[_id] = content
        else:
            pass

    # DO SIMILAR FOR FROMNODE:
    print("Beginning fromnodes...")
    for i in GeoData.drop_duplicates('fromnode').index:
            # Get node ID for filtering duplicates
        _id         = GeoData.iloc[i]['fromnode']

        # Only keep unique observations
        if _id not in nodes:
            # Get attributes if applicable, else get random coordinate in linestring (error usually indicate roundabout as one edge)
            try:
                x        = GeoData.iloc[i]['geometry'].boundary[-1].x
                y        = GeoData.iloc[i]['geometry'].boundary[-1].y
                roadclass = int(GeoData.iloc[i]['funcroadclass'])
                # isBridge = int(GeoData.iloc[i]['isbridge'])
                # isTunnel = int(GeoData.iloc[i]['istunnel'])
                geometry = Point(x,y)
            except Exception as e:
                x        = GeoData.iloc[i]['geometry'].centroid.x
                y        = GeoData.iloc[i]['geometry'].centroid.y
                roadclass = int(GeoData.iloc[i]['funcroadclass'])
                isBridge = int(GeoData.iloc[i]['isbridge'])
                isTunnel = int(GeoData.iloc[i]['istunnel'])
                geometry = Point(x,y)

            # Save and append
            content = {'x':x,'y':y,'osmid':_id, 'roadclass': roadclass, 'isBridge': isBridge, 'isTunnel': isTunnel, 'geometry':geometry}
            nodes[_id] = content
        else:
            pass


    # Get edges into networkx format
    edges = {}
    print("Beginning edges...")
    for i in GeoData.index:
        # Lets keep edges undirected for now (not one way)
        # Get edge ID for filtering duplicates:
        _id                 = GeoData.iloc[i]['linkid']

        # Get only data of edges not already retrieved
        if _id not in edges:
            ref                 = GeoData.iloc[i]['streetname'] 
            funcroadclass       = GeoData.iloc[i]['funcroadclass']
            roadclass           = GeoData.iloc[i]['roadClass']
            isFerry             = GeoData.iloc[i]['isferry']
            isBridge            = GeoData.iloc[i]['isbridge']
            isTunnel            = GeoData.iloc[i]['istunnel']
            speedlim            = GeoData.iloc[i]['speedfw']
            drivetime           = GeoData.iloc[i]['drivetime_fw']
            oneway              = False if GeoData.iloc[i]['oneway'] == "B" else False
            geometry            = GeoData.iloc[i]['geometry']
            u                   = GeoData.iloc[i]['fromnode']
            v                   = GeoData.iloc[i]['tonode']
            key                 = 0

            # linestring_trans = transform(project, GeoData.iloc[i]['geometry'])
            length = GeoData.iloc[i]['length'] - isFerry * GeoData.iloc[i]['length']
            length_weight = length.copy()

            # Estimate length based on speedlimit and drivetime
            # length_estimated = speedlim*drivetime*1000/60

            # Create dictionary of node data:
            content = {'id':_id, 'oneway':oneway, 'ref':ref, 'name':ref, 'funcroadclass':funcroadclass, 'roadclass':roadclass, 'isFerry':isFerry, 'isBridge':isBridge, 'isTunnel':isTunnel, 'speedlim':speedlim, 'drivetime':drivetime, 'length':length, 'length_weight':length_weight, 'geometry':geometry,'u':u, 'v':v, "key": key}

            edges[(u,v,0)] = content
        else:
            pass 

    # Set crs system
    crs = {'init': crs_name}

    # Create for nodes
    nodes_df = gpd.GeoDataFrame(nodes, crs = crs).T
    nodes_df = gpd.GeoDataFrame(
        nodes_df, geometry=nodes_df['geometry'])

    # Create for edges
    edges_df = gpd.GeoDataFrame(edges, crs = crs).T
    edges_df = gpd.GeoDataFrame(
        edges_df, geometry=edges_df['geometry'])


    return nodes_df, edges_df

def get_neighbor_cost(G, source):

    # Get source node elevation 
    s_elevation = G.nodes[source]['elevation']

    neighbors = []

    # For each neighbor, get their elevation
    neighbor_list = [n for n in nx.neighbors(G, source)]
    for n in neighbor_list:
        n_elevation = G.nodes[n]['elevation']

        # Calculate grade based on elevation difference (not direction, as a directed graph)
        # Get length
        try:
            length = G.get_edge_data(source,n)[0]['length']
        except:
            length = G.get_edge_data(source,n)[1]['length']

        # Grade = rise over run
        # If source elevation is higher than neighbor's, grade is negative
        if s_elevation > n_elevation:
            rise = n_elevation - s_elevation
            grade = rise/length
            # print("Edge goes downwards:{}".format(grade))
        # If source elevation is lower than neighbor's, grade is positive
        if s_elevation < n_elevation:
            rise = n_elevation - s_elevation
            grade = rise/length
            # print("Edge goes upwards:{}".format(grade))
        if s_elevation == n_elevation:
            rise = 0 
            grade = 0 
            # print("Edge is flat")
        if length == 0:
            cost = 0

        # If grade is unusually high, 
        cost = calculate_batterycost_single(grade, length)

        neighbors.append((n, cost))

    return neighbors

def shorten_edges_by_cutoff(G, cutoff):
    """
    Function for shortening edges to a specific cutoff threshold. Guarantees that any node can be reached with any range. Divides every edge > cutoff by 2 until no edge surpasses the cutoff value. 
    """

    edges = [e for e in G.edges]
    edges_length = len(edges)
    counter = 0
    new_nodes = []
    edges_shortened = 0

    for e in edges:
        counter += 1
        max_index = max(G.nodes)
        # print("Progess:\t {}".format(counter/edges_length))
        try:
            edge_data = G.edges[e]
            geometry = edge_data['geometry']

            # BEGIN Calculate cost iteratively ================================= 
            source = e[0]
            target = e[1]
            s_elevation = G.nodes[source]['elevation']
            n_elevation = G.nodes[target]['elevation']
            length = edge_data['length']
            if length == 0:
                continue 

            if s_elevation > n_elevation:
                rise = n_elevation - s_elevation
                grade = rise/length
                # print("Edge goes downwards:{}".format(grade))
            # If source elevation is lower than neighbor's, grade is positive
            if s_elevation < n_elevation:
                rise = n_elevation - s_elevation
                grade = rise/length
                # print("Edge goes upwards:{}".format(grade))
            if s_elevation == n_elevation:
                rise = 0 
                grade = 0 

            cost = calculate_batterycost_single(grade, length)
            # END Calculate cost iteratively ================================= 

            # TESTING WITH LOWER CUTOFF VALUE!!! 
            if cost > cutoff:
                # print(cost)
                max_index += 1
                # print("Edge {} surpasses cutoff threshold with edge cost {}".format(e, cost))

                # Retrieve start and end node
                start_node = e[0]
                end_node = e[1]

                # Get coordinates of nodes
                start_node_x = G.nodes[start_node]['x']
                start_node_y = G.nodes[start_node]['y']
                end_node_x = G.nodes[end_node]['x']
                end_node_y = G.nodes[end_node]['y']
                
                # Get roadclass, tunnel and bridge (assuming same as start node)
                roadclass = G.nodes[start_node]['roadclass']
                isbridge = 0
                istunnel = 0
                elevation = G.nodes[start_node]['elevation']

                # Get middle-point between nodes (for new node)
                new_x = (start_node_x + end_node_x)/2
                new_y =  (start_node_y + end_node_y)/2

                # Create Point object
                newpoint = Point([new_x, new_y])

                # Get nearest point along original edge geometry
                np = nearest_points(geometry, newpoint)[0]

                # Add node between start_node and end_node
                # set artificial = True so we know which nodes are inserted into the network
                G.add_node(max_index, x = np.x, y = np.y, osmid = max_index, elevation = elevation, isBridge = isbridge, isTunnel = istunnel, roadclass = roadclass,  geometry = np, artificial = True)
                # Create new geometry between nodes
                first_half = LineString([Point(start_node_x, start_node_y), np])
                second_half = LineString([np, Point(end_node_x, end_node_y)])

                # Create edge between old nodes and new node, delete previous unfeasible edge
                G.add_edge(start_node, max_index,
                id = None, oneway = edge_data['oneway'], ref = edge_data['ref'], 
                name = edge_data['name'], funcroadclass = edge_data['funcroadclass'],
                roadclass = edge_data['roadclass'], isFerry = edge_data['isFerry'], isBridge = isbridge, isTunnel = istunnel, 
                speedlim = edge_data['speedlim'], length = length/2, geometry = first_half, grade = 0, grade_abs = 0)

                G.add_edge(max_index, end_node,
                id = None, oneway = edge_data['oneway'], ref = edge_data['ref'], 
                name = edge_data['name'], funcroadclass = edge_data['funcroadclass'],
                roadclass = edge_data['roadclass'], isFerry = edge_data['isFerry'], isBridge = isbridge, isTunnel = istunnel, 
                speedlim = edge_data['speedlim'], length = length/2, geometry = second_half, grade = 0, grade_abs = 0)

                # print("Edge added...")

                G.remove_edge(e[0], e[1])
                new_nodes.append(max_index)
                edges_shortened += 1

        except KeyError as KE:
            # print("ERROR:\t{}".format(e))
            pass

    if edges_shortened > 0:
        print("Performing recursion...")
        shorten_edges_by_cutoff(G = G, cutoff = cutoff)
    else:
        print("No condition satisfied. Recursion ended and function completed.")
        pass  

def dijkstra_cutoff(graph, Q, source, cutoff, weight = 'battery_cost'):
    """
    Function for constructing a reachability graph from a given source node. Q is the list of nodes present in the graph 
    Parameters
    ----------
    graph : NetworkX graph object
    Q     : set of nodes, e.g. set(n for n in graph.nodes())
    source: int, source node in graph
    cutoff : int, default cost threshold
    weight : string, edge weight to be evaluated, default = 'battery_cost
    """
    Q2 = Q.copy()

    # Create dict for distances
    dist = {}
    # Keep track of visisted nodes
    visited = set()

    # Set distance to infinity for every node except source, which is 0:
    dist = {n: float("inf") for n in Q}
    dist[source] = 0

    while Q2:
        dist_u = float("inf")
        u = None

        # Return the node with the shortest cost from source
        for n in Q2:
            if dist[n] < dist_u and n not in visited:
                dist_u = dist[n]
                u = n



        # If no condition is fulfilled, we are done
        if u is None:
            reachability = {k:v for k,v in dist.items() if v <= cutoff}
            reachability.pop(source)
            return reachability

        # Add u to visisted and remove from Q
        visited.add(u)
        Q2.remove(u)


        # Retrieve neighbors of u
        neighbors = get_neighbor_cost(graph, u)
        # For each neighbor of u:
        for (neighbor, cost) in neighbors:
            total_dist = cost + dist_u

            # If end up travelling further than the cutoff, save and check next neighbor
            if cutoff is not None:
                if total_dist >= cutoff:
                    continue

            if total_dist < dist[neighbor]:
                dist[neighbor] = total_dist             

def get_node_attributes(graph, reachable_nodes):
    attributes = {}
    data_target = [n for n in graph.nodes()]
    for n in data_target:
        try:
            data = graph.nodes[n]
            x = data['x']
            y = data['y']
            _id = data['osmid']
            roadclass = data['roadclass']
            attributes[n] = {"x":x, "y":y, "roadclass":roadclass, "osmid":_id}
        except Exception as e:
            data = graph.nodes[n]
            x = data['x']
            y = data['y']
            roadclass = data['roadclass']
            _id = data['id']
            attributes[n] = {"x":x, "y":y, "roadclass":roadclass, "osmid":_id}

    return attributes

def get_inverted_latlon(graph, nodes = None):
    pos = {}
    if nodes is not None:
        for i in nodes:
            data = graph.nodes[i]
            x = float(data['x'])
            y = float(data['y'])
            pos[i] = (x,y)
    else:
        nodes = [n for n in graph.nodes()]
        for i in nodes:
            data = graph.nodes[i]
            x = float(data['x'])
            y = float(data['y'])
            pos[i] = (x,y)

    return pos

def unweighted_dijkstra_cutoff(graph, Q, source, cutoff, weight = 'battery_cost'):
    """
    Function for constructing a reachability graph from a given source node. Q is the list of nodes present in the graph 
    Parameters
    ----------
    graph : NetworkX graph object
    Q     : List of nodes, e.g. [n for n in graph.nodes()]
    source: int, source node in graph
    cutoff : int, default cost threshold
    weight : string, edge weight to be evaluated, default = 'battery_cost
    """
    Q2 = Q.copy()

    # Create dict for distances
    dist = {}
    # Keep track of visisted nodes
    visited = set()

    # Set distance to infinity for every node except source, which is 0:
    dist = {n: float("inf") for n in Q}
    dist[source] = 0

    while Q2:
        u_dist = float("inf")
        u = None

        # Return the node with the shortest path from source
        # for n in Q????? 
        for n in Q2:
            if dist[n] < u_dist and n not in visited:
                u_dist = dist[n]
                u = n

        # If no condition is fulfilled, we are done
        if u is None:
            # elapsed_time = time.time() - start
            # print("Elapsed time for {} cutoff:\t".format(cutoff), elapsed_time)
            del dist[source]
            reachability = [k for k,v in dist.items() if v <= cutoff]
            return reachability

        # Add u to visisted and remove from Q
        visited.add(u)
        Q2.remove(u)

        # Retrieve neighbors of u
        neighbors = get_neighbor_cost(graph, u, weight = 'battery_cost')
        # print('neighbors:\t',neighbors)
        # For each neighbor of u:
        for (neighbor, cost) in neighbors:
            total_dist = cost + u_dist

            # If end up travelling further than the cutoff, save and check next neighbor
            if cutoff is not None:
                if total_dist >= cutoff:
                    continue

            if total_dist < dist[neighbor]:
                dist[neighbor] = total_dist    

def graph_from_reachability_graph(graph, reachable_nodes, source_node, weight):
    # Get nodes into list
    g = nx.subgraph(graph, [n for n in reachable_nodes])

    # Get edge weight
    vals = []
    for key, val in reachable_nodes.items():
        vals.append((source_node, key, {weight:val}))
    
    # Create empty copy of g (no edge data)
    g = nx.create_empty_copy(g)

    # Add edge data
    g.add_weighted_edges_from(vals, weight = weight)

    return g

def compare_outputs(g, g2, cutoff, save_output = None):
    pos1 = get_inverted_latlon(g)
    pos2 = get_inverted_latlon(g2)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10,5))
    fig.suptitle("Comparison of Dijkstra with cutoff and ego graph with {} cutoff".format(cutoff))

    # Plot base layer
    # muni_geo.plot(ax = ax1)
    # muni_geo.plot(ax = ax2)

    # # Plot graphs
    nx.draw(g, pos = pos1, ax = ax1, node_size = 0.1, linewidths = 0.1,  node_color = 'purple', edge_color = 'red', width   = 0.1)
    nx.draw(g2, pos = pos2, ax = ax2, node_size = 0.1, linewidths = 0.1, node_color = 'purple', edge_color = 'red', width   = 0.1)
    
    # # Set ax limits based on maximum and minimum positions
    minx, maxx = min([v[0] for k,v in pos2.items()]), max([v[0] for k,v in pos2.items()])
    miny, maxy = min([v[1] for k,v in pos2.items()]), max([v[1] for k,v in pos2.items()])

    ax1.set_xlim([minx - 0.01, maxx + 0.01])
    ax1.set_ylim([miny - 0.01, maxy + 0.01])
    ax2.set_xlim([minx - 0.01, maxx + 0.01])
    ax2.set_ylim([miny - 0.01, maxy + 0.01])

    # Save figure if
    if save_output is not None:
        plt.savefig(save_output, dpi = 500, bbox_inches = 'tight')

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def get_k_neighbors(G, start, k):
    k_neighbors = set([start])
    for depth in range(k):
        k_neighbors = set((nbr for n in k_neighbors for nbr in G[n]))
    k_neighbors.remove(start)
    return k_neighbors

def is_k_dominating_set(G, D, k):

    # loop through the nodes in the complement of S and determine
    # if they are adjacent to atleast k nodes in S
    others = set(G.nodes()).difference(D)
    for v in others:
        if len(set(G.neighbors(v)).intersection(D)) < k:
            return False
    # if the above loop completes, nbunch is a k-dominating set
    return True

def randomized_k_dominating(K, k):
    degrees = [v for n,v in K.degree()]
    d = np.mean(degrees)
    D = max(degrees)
    d_prime = d - k + 1
    binomial_comp  =  scipy.special.binom(d, k-1)
    denominator =  (binomial_comp*(1+d_prime))**(1/d_prime)

    p = 1-(1/denominator)

    nodes = {n for n in K.nodes()}
    # initialize set A = {0}
    A = set()
    for node in nodes:
        isInA = np.random.choice(a = [True, False], p = [p, 1-p])
        # This forms a subset A in V
        if isInA == True:
            A.add(node)
            
    B = set()
    for node in nodes.difference(A):
        neighbors = {n for n in K.neighbors(node)}

        neighbors_in_a = neighbors.intersection(A)
        if len(neighbors_in_a) < k:
            B.add(node)

    D = A.union(B)
    # print(len(D))

    # Reduce cardinality of set
    D = minimal_k_dominating(K, D, k)

    return D

def minimal_k_dominating(K, D, k):
    # Sort vertices by the number of neighbors not in D they have
    nodes = {n for n in D}
    neighbors_not_in_D = {}
    for n in nodes:
        n_neighbors = len({m for m in K.neighbors(n)}.difference(D))
        neighbors_not_in_D[n] = n_neighbors

    sorted_dict = {k: v for k, v in sorted(neighbors_not_in_D.items(), key=lambda item: item[1], reverse=True)}
    L = [k for k in sorted_dict]

    counter = 0
    length = len(L)
    # print("Original dominating set:\t{}".format(D))
    for v in L:
        counter += 1
        print(round(counter/length,5), end = '\r')
        v_set = set([v])

        if is_k_dominating_set(K, D.difference(v_set), k):
            # print("D is k-dominating in K without node {}... Removing {}".format(v,v))
            # print("Removing {} from D".format(v))
            D.remove(v)
            # print(D)
    return D

def greedy_redundant_removal(G, D, k):
    k_neighbors = {}
    # Sort vertices in ascending order of size of their k-neighbor set
    counter = 0
    length = len(G.nodes)
    for i in G.nodes():
        counter += 1
        print("{}\tbuilding graphs".format(counter/length))
        k_graph = get_k_neighbors(G, i, k)
        k_neighbors[i] = len(k_graph)
    D_sorted = {k: v for k, v in sorted(k_neighbors.items(), key=lambda item: item[1], reverse=False)}

    k1 = int(0.5*(k+1))
    k2 = int(k-k1)

    counter = 0 
    length = len(D_sorted)
    for v in D_sorted:
        counter += 1
        print("{}\tlooping".format(counter/length))
        v_set = {v}
        S = True
        Nvk =  get_k_neighbors(G, v, k)
        for u in Nvk:
            T = False
            Nuk1 = get_k_neighbors(G, u, k1)
            for w in D.difference(v_set):
                Nwk2 =  get_k_neighbors(G, w, k2) 
                if not Nwk2.intersection(Nuk1):
                    T = True

            if T == False:
                S = False
        if S == True:
            D = D.difference(v_set) 
    return D

def greedy_heuristic_1(G, k):
    # Sort vertices in descending order of degree
    nodes = dict(G.degree)
    nodes = {k: v for k, v in sorted(nodes.items(), key=lambda item: item[1], reverse=True)}

    isCovered = {}
    for v in nodes:
        isCovered[v] = False
    D = set()
    for v in nodes:
        v_set = set()
        v_set.add(v)
        if isCovered[v] == False:
            D = D.union(v_set)
            k_neighbors =  get_k_neighbors(G, v, k)
            for u in k_neighbors:
                isCovered[u] = True
    
    return D

def greedy_heuristic_2(G, k, theta):
    # Sort vertices in descending order of degree
    nodes = dict(G.degree)
    nodes = {k: v for k, v in sorted(nodes.items(), key=lambda item: item[1], reverse=True)}
    isCovered = {}
    for v in nodes:
        isCovered[v] = False
    D = set()
    for v in nodes:
        v_set = set()
        v_set.add(v)
        k_neighbors = get_k_neighbors(G, v, k) 
        uncovered_k_neighbors = [n for n in k_neighbors if isCovered[n] == False]
        if isCovered[v] == False or len(uncovered_k_neighbors) >= theta:
            D = D.union(v_set)
            for u in k_neighbors:
                isCovered[u] = True
    return D

def roadclass_minimal_k_dominating(K, D, k):

    # Sort vertices by a product of roadclass and neighbors not in D 
    nodes_list = {n for n in D}
    roadclass_dict = {}
    #neighbors_not_in_D = {}
    for n in nodes_list:
        n_neighbors = len({m for m in K.neighbors(n) if m not in D})
        roadclass = int(K.nodes[n]['roadclass'])
        roadclass_dict[n] = roadclass 
        #product = n_neighbors * (5-roadclass)
        #product_dict[n] = product 
        # neighbors_not_in_D[n] = n_neighbors

    sorted_dict = {k: v for k, v in sorted(roadclass_dict.items(), key=lambda item: item[1], reverse=True)}
    L = {k for k in sorted_dict}

    counter = 0
    length = len(L)
    # print("Original dominating set:\t{}".format(D))
    for v in L:
        counter += 1
        print(round(counter/length,5), end = '\r')
        v_set = set()
        v_set.add(v)
        
        if is_k_dominating_cg2018(K, D.difference(v_set), k):
            # print("Removing {} from D".format(v))
            D = D.difference(v_set)
            # print(D)
    return D

def calculate_batterycost(G):
    """
    Calculates the battery cost of traversing an edge. Formula is length * (1+grade). Ignores ferries.
    """

    # Coefficients indiciate consumption in kWh per KILOMETER
    # Numbers from https://www.sciencedirect.com/science/article/pii/S1361920917303887 
    coefficients = [-0.332, -0.217, -0.148, -0.121, -0.073, 0.085, 0.152, 0.203, 0.306, 0.358, 0.552]
    gradients = [-0.09, -0.07, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11]
    const = 0.372
    coeffs = {}
    for i in enumerate(gradients):
        _index = i[0]
        value = i[1]
        coeffs[value] = coefficients[_index]/1000 # Get coefficient value and divide by 1000 to retrieve coefficient in meters


    lengths = nx.get_edge_attributes(G, 'length')
    grades = nx.get_edge_attributes(G, 'grade')
    # isferry = nx.get_edge_attributes(G, 'isFerry')

    costs = {}

    # Iterate through edges and calculate battery cost
    for key, length in lengths.items():
        # If not ferry:
        if length != 0:
            grade = grades[key]
            
            # Messy...
            kwh_cost = None
            if grade < gradients[0]:
                kwh_cost = coefficients[0]
            if gradients[0] <= grade < gradients[1]:
                kwh_cost = coefficients[1]
            if gradients[1] <= grade < gradients[2]:
                kwh_cost = coefficients[2]
            if gradients[2] <= grade < gradients[3]:
                kwh_cost = coefficients[3]
            if gradients[3] <= grade < gradients[4]:
                kwh_cost = coefficients[4]
            if gradients[4] <= grade < gradients[5]:
                kwh_cost = coefficients[5]
            if gradients[5] <= grade < gradients[6]:
                kwh_cost = coefficients[6]
            if gradients[6] <= grade < gradients[7]:
                kwh_cost = coefficients[6]
            if gradients[7] <= grade < gradients[8]:
                kwh_cost = coefficients[8]
            if grade > gradients[9]:
                kwh_cost = coefficients[9]

            cost = (const + kwh_cost) * length
            costs[key] = cost
        if length == 0:
            cost = 0

    # Convert nan in costs to 0
    for key, cost in costs.items():
        if np.isnan(cost):
            costs[key] = 0

    # Set the edge attribute from dictionary created in for loop
    nx.set_edge_attributes(G, costs, 'battery_cost')

def calculate_batterycost_single(grade, length):
    # Coefficients indiciate consumption in kWh per KILOMETER
    # Numbers from https://www.sciencedirect.com/science/article/pii/S1361920917303887 
    coefficients = [-0.332, -0.217, -0.148, -0.121, -0.073, 0.085, 0.152, 0.203, 0.306, 0.358, 0.552]
    gradients = [-0.09, -0.07, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11]
    const = 0.372
    coeffs = {}
    for i in enumerate(gradients):
        _index = i[0]
        value = i[1]
        coeffs[value] = coefficients[_index]

    cost = None
    # If not ferry:
    if length != 0:
        
        # Messy...
        kwh_cost = None

        if grade == 0:
            kwh_cost = 0

        kwh_cost = None
        if grade < gradients[0]:                    # Up to -9%
            kwh_cost = coefficients[0]      
        if gradients[0] <= grade < gradients[1]:    # -9 to -7%
            kwh_cost = coefficients[1]  
        if gradients[1] <= grade < gradients[2]:     # -7 to -5%
            kwh_cost = coefficients[2]
        if gradients[2] <= grade < gradients[3]:     # -5 to -3% 
            kwh_cost = coefficients[3]
        if gradients[3] <= grade < gradients[4]:     # -3 to -1%
            kwh_cost = coefficients[4]
        if gradients[4] <= grade < gradients[5]:     # 1 to 3% 
            kwh_cost = coefficients[5]
        if gradients[5] <= grade < gradients[6]:    # 3 to 5%
            kwh_cost = coefficients[6]
        if gradients[6] <= grade < gradients[7]:     # 5 to 7%
            kwh_cost = coefficients[7]
        if gradients[7] <= grade < gradients[8]:    # 7 to 9%
            kwh_cost = coefficients[8]
        if gradients[8] <= grade <= gradients[9]:   # 9 to 11%
            kwh_cost = coefficients[9]
        if grade > gradients[9]:                    # More than 11%
            kwh_cost = coefficients[10]

        # print(kwh_cost)
        cost = (const + kwh_cost)/1000 * length
    if length == 0:
        cost = 0
    # print("Gradient is {} and length is {}. Cost is {}".format(grade, length, cost))
    return cost

def outlier_aware_hist(data, lower=None, upper=None):
    if not lower or lower < min(data):
        lower = min(data)
        lower_outliers = False
    else:
        lower_outliers = True

    if not upper or upper > max(data):
        upper = max(data)
        upper_outliers = False
    else:
        upper_outliers = True

    n, bins, patches = plt.hist(data, range=(lower, upper), bins=25, edgecolor = 'white')

    if lower_outliers:
        n_lower_outliers = (data < lower).sum()
        patches[0].set_height(patches[0].get_height() + n_lower_outliers)
        patches[0].set_facecolor('c')
        patches[0].set_label('Lower outliers: ({:.2f}, {:.2f})'.format(min(data), lower))

    if upper_outliers:
        n_upper_outliers = (data > upper).sum()
        patches[-1].set_height(patches[-1].get_height() + n_upper_outliers)
        patches[-1].set_facecolor('m')
        patches[-1].set_label('Upper outliers: ({:.2f}, {:.2f})'.format(upper, max(data)))

    if lower_outliers or upper_outliers:
        plt.legend(prop={'size': 12})

def mad(data):
    median = np.median(data)
    diff = np.abs(data - median)
    mad = np.median(diff)
    return mad

def calculate_bounds(data, z_thresh=15):
    MAD = mad(data)
    median = np.median(data)
    const = z_thresh * MAD / 0.6745
    return (median - const, median + const)

def compute_required_reach(graph, Q, source, D):
    """
    Function for constructing a reachability graph from a given source node. Q is the list of nodes present in the graph 
    Parameters
    ----------
    graph : NetworkX graph object
    Q     : set of nodes, e.g. set(n for n in graph.nodes())
    source: int, source node in graph
    cutoff : int, default cost threshold
    weight : string, edge weight to be evaluated, default = 'battery_cost
    """
    Q2 = Q.copy()

    # Create dict for distances
    dist = {}
    # Keep track of visisted nodes
    visited = set()

    # Set distance to infinity for every node except source, which is 0:
    dist = {n: float("inf") for n in Q}
    dist[source] = 0

    while Q2:
        dist_u = float("inf")
        u = None

        # Return the node with the shortest cost from source
        for n in Q2:
            if dist[n] < dist_u and n not in visited:
                dist_u = dist[n]
                u = n

        # If no condition is fulfilled, we are done
        if u is None:
            reachability = {k:v for k,v in dist.items() if v <= cutoff}
            reachability.pop(source)
            return reachability

        # Add u to visisted and remove from Q
        visited.add(u)
        Q2.remove(u)

        # Retrieve neighbors of u
        neighbors = get_neighbor_cost(graph, u)
        # For each neighbor of u:
        for (neighbor, cost) in neighbors:
            total_dist = cost + dist_u

            # If the neighbor is in the candidate solution D:
            if neighbor in D:
                return total_dist

            if total_dist < dist[neighbor]:
                dist[neighbor] = total_dist    

def construct_reachability_graph(G, cutoff):

    enumerated_nodes = list(enumerate(G.nodes))
    indices = {n[1]:n[0] for n in enumerated_nodes}

    G = nx.relabel_nodes(G, indices)

    # Create empty adjacency matrix
    g = nx.adjacency_matrix(nx.create_empty_copy(G))
    g = sparse.lil_matrix(g)
    _length = len(G.nodes)
    for i in list(G.nodes):
        # print(i/_length, end = '\r')
        # Retrieve range neighborhood of vertex
        source = i
        Q = set(G.nodes)
        nbrhood = dijkstra_cutoff(G, Q, source, cutoff)
        # print("Dijkstra time:\t",end-start)

        for i in nbrhood:
            g[source,i] += nbrhood[i]

    return g

def plot_algorithm_step(G, colors, step, pos):
    color_map = [v for k,v in colors.items()]
    plt.figure(figsize = (5,5))
    nx.draw(G, node_size = 1000, width = 0.5, node_color = color_map, pos = pos, linewidths = 1, with_labels = True)
    ax = plt.gca()
    ax.collections[0].set_edgecolor("black") 
    plt.savefig('gkcds_steps/step_{}.png'.format(step))
    plt.close()

def G_CDS(G, pos = None):
    step = 0
    count = len(G)

    dom_col = 'red'
    cov_col = 'grey'
    un_col = 'white'


    neighborhoods = {n: set(G.neighbors(n)) for n in G.nodes}
    colors = {n: un_col for n in G.nodes} # All nodes are initially white
    degrees = dict(G.degree)

    # pos = get_inverted_latlon(G)
    if pos:
        plot_algorithm_step(G, colors, step, pos)

    v = max(degrees, key = degrees.get)
    colors[v] = dom_col
    count -= 1

    for w in neighborhoods[v]:
        if v in neighborhoods[w]:
            neighborhoods[w].remove(v)
            # print("Removed {} from neighborhood of {}".format(v,w))
            colors[w] = cov_col
            count -= 1
    
    for u,c in colors.items():
        if c == cov_col:
            for w in neighborhoods[u]:
                if colors[w] == cov_col:
                    if u in neighborhoods[w]:
                        neighborhoods[w].remove(u)
                        # print("Removed {} from neighborhood of {}".format(u,w))

    step += 1
    if pos:
        plot_algorithm_step(G, colors, step, pos)

    while count > 0:
        # print(count)
        step += 1
        # Select gray node v with largest number of white neighbors among all grey nodes
        degrees = {}
        for v,c in colors.items():
            if c == cov_col:
                v_nbrs = set(n for n in G.neighbors(v) if colors[n] == un_col)
                degrees[v] = len(v_nbrs)

        v = max(degrees, key = degrees.get)
        colors[v] = dom_col
        # count -= 1

        for w in neighborhoods[v]:
            if v in neighborhoods[w]:
                neighborhoods[w].remove(v)
                colors[w] = cov_col
                count -= 1
        
        for u,c in colors.items():
            if c == cov_col:
                for w in neighborhoods[u]:
                    if colors[w] == cov_col:
                        if u in neighborhoods[w]:
                            neighborhoods[w].remove(u)
        if pos:
            plot_algorithm_step(G, colors, step, pos)

    # D = set(k for k,v in colors.items() if v == dom_col)
    # covered = set(k for k,v in colors.items() if v == cov_col)
    

    return colors

def P_CDS(G, D):

    D2 = D.copy()

    for v in D:
        if grinpy.is_connected_dominating_set(G, D.difference({v})):
            D2.remove(v)
            print("Removed {}".format(v))
        else:
            pass

    return D2

def P_kCDS(G, D, k):

    D2 = D.copy()

    for v in D:
        if grinpy.is_connected_k_dominating_set(G, D.difference({v}), k):
            D2.remove(v)
            print("Removed {}".format(v))
        else:
            pass

    return D2

def G_kCDS(G, k, pos = None):
    
    # if k < 2 or type(k) != int :
    #     raise ValueError("k must be integer higher than 1")

    step = 0
    count = len(G.nodes)

    # Instantiate neighborhood
    neighborhood = {n: set(G.neighbors(n)) for n in G.nodes}

    # Instantiate colors
    colors = {n: 'white' for n in G.nodes}

    if pos:
        plot_algorithm_step(G, colors, step, pos)
        
    # Instantiate degree
    degrees = dict(G.degree)

    # Instantiate candidate nodes
    C = set()
    D = set()

    # Choose max degree node
    v = max(degrees, key = degrees.get)
    colors[v] = 'red'   # set red
    D.add(v)

    # Color neighbors yellow
    for u in neighborhood[v]:
        if colors[u] == 'white':
            colors[u] = 'yellow'
            C.add(u)

        # If a node has two red neighbors AND that node is not red, set it as green
        # DOES NOT USUALLY APPLY IN FIRST STEPS
        if (len(neighborhood[u].intersection(D)) >=k and colors[u] != 'red'):
            colors[u] = 'green'
            count -= 1

    step += 1
    if pos:
        plot_algorithm_step(G, colors, step, pos)

    # While the set is not connected k-dominating:
    while any(c == 'white' or c == 'yellow' for n,c in colors.items()):
        # print(step)
        step += 1

        # Retrieve degrees
        degrees = {}
        for c in C:
            degrees[c] = len(set(n for n in neighborhood[c] if (colors[n] != 'red' and colors[n] != 'green')))

        # If remaining degrees are all zero, remove all green nodes from degrees
        if sum(degrees.values()) == 0:
            # print("DEG B4: ", degrees)
            degrees = {n:degrees[n] for n,c in colors.items() if c == 'yellow' and n in degrees}
            # print(degrees)

        # print(step)
        # print(degrees)
        # Select node from candidate pool with max degree
        v = max(degrees, key = degrees.get)

        # Color node red
        colors[v] = 'red'
        C.remove(v) # Remove it from candidate pool
        D.add(v)    # Add it to dominating set pool

        # For each neighbor of v: if that node is white (e.g not green or red) color it yellow and add it as candidate
        for u in neighborhood[v]:
            if colors[u] == 'white':
                colors[u] = 'yellow'
                C.add(u)
            
            # If a node has two red neighbors AND that node is not red, set it as green
            if (len(neighborhood[u].intersection(D)) >=k and colors[u] != 'red'):
                colors[u] = 'green'
                count -= 1

        if pos:
            plot_algorithm_step(G, colors, step, pos)
    return colors

def G_kCDS_adj(A, degrees, neighborhood, k):

    start = time.time()
    # Instantiate count and step
    step = 0
    count = A.shape[0]

    # Instantiate colors
    colors = {n: 'white' for n in range(count)}
    # print("\tColor instantiated...")

    # Instantiate candidate nodes
    C = set()
    D = set()

    # Choose maximum degree node
    v = max(degrees, key = degrees.get)
    colors[v] = 'red'   # set red
    D.add(v)
    count -= 1

    # Color neighbors yellow
    for u in neighborhood[v]:
        if colors[u] == 'white':
            colors[u] = 'yellow'
            C.add(u)

        # If a node has two red neighbors AND that node is not red, set it as green
        # DOES NOT USUALLY APPLY IN FIRST STEPS
        if (len(neighborhood[u].intersection(D)) >=k and colors[u] != 'red'):
            colors[u] = 'green'
            count -= 1

    # While set is not k-dominating
    while any(c == 'white' or c == 'yellow' for n,c in colors.items()):
        step += 1

        # Retrieve degrees
        subdegrees = {}
        for c in C:
            subdegrees[c] = len(set(n for n in neighborhood[c] if (colors[n] != 'red' and colors[n] != 'green')))

        # If remaining degrees are all zero, remove all green nodes from degrees and color remaining yellow (must be)
        if sum(subdegrees.values()) == 0:
            # Color remaining yellow
            remaining = set(n for n,c in colors.items() if c == 'yellow')
            for i in remaining:
                colors[i] = 'red'
            end = time.time()
            return colors 

        # Select node from candidate pool with max degree
        v = max(subdegrees, key = subdegrees.get)

        # Color node red
        colors[v] = 'red'
        C.remove(v) # Remove it from candidate pool
        D.add(v)    # Add it to dominating set pool

        # For each neighbor of v: if that node is white (e.g not green or red) color it yellow and add it as candidate
        for u in neighborhood[v]:
            if colors[u] == 'white':
                colors[u] = 'yellow'
                C.add(u)
            
            # If a node has k red neighbors AND that node is not red, set it as green
            if (len(neighborhood[u].intersection(D)) >=k and colors[u] != 'red'):
                colors[u] = 'green'

    end = time.time()
    return colors

def create_undirected_matrix_from_adjacency_matrix(A, debug_faulty_edge = False):

    n = A.shape[0]
    B = sparse.lil_matrix(A.shape)

    checked = set()

    # Iterate through each column (node)
    for v in range(n):

        # print(v)
        a = A[:,v]

        # For each node, retrieve its neighbor
        nbrs = list(a.nonzero()[0])

        # For each neighbor, retrieve BOTH costs
        for u in nbrs:

            # Register edge pair
            edge = (v,u)

            # If edge pair already checked
            # THIS DOES NOT WORK, RE-ASSES!!! 
            if edge in checked or edge[::-1] in checked:
                # print("{} already checked".format(edge))
                continue

        
            costs = (A[v,u], A[u,v])
            c = max(costs)

            # If 0 is in the costs, edge is only one way and should not be added
            if 0 in costs:
                checked.add(edge)
                checked.add(edge[::-1])
                continue
            
            B[v,u] = c
            B[u,v] = c

            checked.add(edge)
            checked.add(edge[::-1])
        

    
    B = sparse.csr_matrix(B)

    # Force fix one faulty edge
    if (B[8291,496] >= 20 or B[496, 8291] >= 20) and debug_faulty_edge == True:
        B[8291,496] = 19.999
        B[496, 8291] = 19.999

    return B

def create_undirected_graph_from_adjacency_matrix(A):
    # Construct undirected graph from adjacency matrix, keeping only conservative edges
    # This means that if an edge goes both ways, the highest cost should represent the edge weight
    # If the edge is not reciprocal, then the edge is disregarded

    nodes = set(range(A.shape[0]))
    edges = {}
    to_check = set(range(A.shape[0]))
    checked = set()

    # Iterate through each node
    for i in to_check:
        print(i)
        b = A[:,i]

        # Retrieve neighbors
        nbrs = b.nonzero()[0]

        # For each neighbor, check the edge cost AND if edge is reciprocal
        for j in nbrs:
            
            # If edge pair already constructed
            if (i,j) in edges or (j,i) in edges:
                continue

            costs = [A[i,j], A[j,i]]
            _max = max(costs)

            # If cost is not reciprocal, set edge to 0
            if any([0, 0.0]) in costs:
                pass 
            # Else, choose max cost
            else:
                edges[(i,j)] = _max
                checked.add(j)

    # Fix that one edge for needs to be forced
    edges[(8291, 496)] = 19.999
    return nodes, edges

def is_connected_k_dominating_adj(A, neighborhood, D, k):

    count = A.shape[0]
    nodes = set(range(count))

    # Check if set is dominating
    others = nodes.difference(D)
    for v in others:
        if len(neighborhood[v].intersection(D)) < k:
            return False
    
    # Check for connectivity in adj.matrix
    G = nx.from_scipy_sparse_matrix(A[np.ix_(list(D), list(D))])
    if grinpy.is_connected(G):
        return True
    else:
        return False

def prune_gckds_adj(A, neighborhood, D, degrees, k):
    # For each red node, check if the set is dominating without it
    B = D.copy()

    # Sort vertices in D based on their degree
    d = [k for k, v in sorted(degrees.items(), key=lambda item: item[1], reverse=False)]
    D = [n for n in d if n in D]

    for v in D:
        print("\t", v, end = '\r')
        if is_connected_k_dominating_adj(A, neighborhood, B.difference({v}), k):
            print("\t set is dominating without {}".format(v))
            B.remove(v)
    return B

def get_set_metrics(pickled_sets, n_sets = 12, prunes = None):

    # Rename pickled_sets
    sets = pickled_sets

    if prunes:
        R = [20,30,40]
        set_params = []
        for r in R:
            k = [1,2,3,4]
            for i in k:
                set_params.append([r,i])
        categories = [(n[0], n[1]) for n in set_params]
        data = pd.DataFrame(columns=["range", "k", "n"])

        count = 0
        index = ["range", "k", "n"]
        for i in categories:

            d = [i[0], i[1], len(sets[count])]
            data = data.append(pd.Series(d, index = index), ignore_index=True)
            count += 1

        # Instantiate sets of global data collectors
        dists = []
        neighbors = []
        locs = []


        # For each set in the collection, compute values:
        counter = 0
        for s in sets:

            to_iterate = [n[0] for n in enumerate(G.nodes) if n[0] not in s]

            if counter < 4:
                A = scipy.sparse.load_npz('data/Reachability_03_20kwh_UNDIRECTED.npz')
            if counter >= 4:
                A = scipy.sparse.load_npz('data/Reachability_03_30kwh_UNDIRECTED.npz')
            if counter >= 8:
                A = scipy.sparse.load_npz('data/Reachability_03_40kwh_UNDIRECTED.npz')

            # Statistics of sets:
            all_dists = []
            n_neighbors = []
            coords = []


            # For each row, compute:
            for i in to_iterate:
                a = A[i,list(s)]
                a_data = a.data

                # Store distance data
                for j in a_data:
                    all_dists.append(j)

                # Store neighbor a_data
                n = len(a_data)
                n_neighbors.append(n)

                # Store coordinates
                x = G.nodes[i]
                y = G.nodes[i]
                coords.append((x,y))

            
            # Append to full collection
            dists.append(all_dists)
            neighbors.append(n_neighbors)
            counter += 1
            locs.append(coords)

        data['category'] = list(zip(data.range, data.k))
        data2 = pd.DataFrame(data['category'].drop_duplicates())

        # Map distances to each category
        data2['Mean dist.'] = [round(np.mean(i), 3) for i in dists]
        data2['Median dist.'] = [round(np.median(i),3) for i in dists]
        data2['Max dist.'] = [round(np.max(i),3) for i in dists]
        data2['Min dist.'] = [round(np.min(i),4) for i in dists]
        data2['$|D|_1$'] = [p[0] for p in prunes]
        data2['$|D|_2$'] = [len(i) for i in sets]


        # Map n neighbors to each category
        data2['Mean nbrs.'] = [round(np.mean(i),3) for i in neighbors]
        data2['Median nbrs.'] = [int(np.median(i)) for i in neighbors]
        data2['St.dev.'] = [np.round(np.std(i),3) for i in neighbors]
        data2['Max nbrs.'] = [np.max(i) for i in neighbors]
        data2['Min nbrs.'] = [np.min(i) for i in neighbors]

        distances = data2[['category', 'Mean dist.', 'Median dist.', 'Min dist.', '$|D|_1$', '$|D|_2$']]
        nbrs = data2[['category', 'Mean nbrs.', 'Median nbrs.','St.dev.', 'Min nbrs.', 'Max nbrs.']]

        print("========== Distance CkDS metrics ==========")
        print(distances.to_latex(index = False))

        print("========== Neighbor CkDS metrics ==========")
        print(nbrs.to_latex(index = False))

    # If no "prunes" is passed, set should be k-dominating (not connected)
    else:
        R = [10,15,20]
        set_params = []
        for r in R:
            k = [1,2,3,4]
            for i in k:
                set_params.append([r,i])

        categories = [(n[0], n[1]) for n in set_params]
        data = pd.DataFrame(columns=["range", "k", "n"])


        count = 0
        index = ["range", "k", "n"]
        for i in categories:

            d = [i[0], i[1], len(sets[count])]
            data = data.append(pd.Series(d, index = index), ignore_index=True)
            count += 1

        # Instantiate sets of global data collectors
        dists = []
        neighbors = []
        locs = []


        # For each set in the mins collection, compute values:
        counter = 0
        for s in sets:

            to_iterate = [n[0] for n in enumerate(G.nodes) if n[0] not in s]

            if counter < 4:
                A = scipy.sparse.load_npz('data/Reachability_03_10kwh_UNDIRECTED.npz')
            if counter >= 4:
                A = scipy.sparse.load_npz('data/Reachability_03_15kwh_UNDIRECTED.npz')
            if counter >= 8:
                A = scipy.sparse.load_npz('data/Reachability_03_20kwh_UNDIRECTED.npz')

            # Statistics of sets:
            all_dists = []
            n_neighbors = []
            coords = []


            # For each row, compute:
            for i in to_iterate:
                a = A[i,list(s)]
                a_data = a.data

                # Store distance data
                for j in a_data:
                    all_dists.append(j)

                # Store neighbor a_data
                n = len(a_data)
                n_neighbors.append(n)

                # Store coordinates
                x = G.nodes[i]
                y = G.nodes[i]
                coords.append((x,y))

            
            # Append to full collection
            dists.append(all_dists)
            neighbors.append(n_neighbors)
            counter += 1
            locs.append(coords)

            data['category'] = list(zip(data.range, data.k))
            data2 = pd.DataFrame(data['category'].drop_duplicates())

            # Map distances to each category
            data2['Mean dist.'] = [round(np.mean(i), 3) for i in dists]
            data2['Median dist.'] = [round(np.median(i),3) for i in dists]
            data2['Max dist.'] = [round(np.max(i),3) for i in dists]
            data2['Min dist.'] = [round(np.min(i),4) for i in dists]
            data2['$|D|$'] = [len(i) for i in sets]

            # Map n neighbors to each category
            data2['Mean nbrs.'] = [round(np.mean(i),3) for i in neighbors]
            data2['Median nbrs.'] = [int(np.median(i)) for i in neighbors]
            data2['St.dev.'] = [np.round(np.std(i),3) for i in neighbors]
            data2['Max nbrs.'] = [np.max(i) for i in neighbors]
            data2['Min nbrs.'] = [np.min(i) for i in neighbors]

            distances = data2[['category', 'Mean dist.', 'Median dist.', 'Min dist.', '$|D|$']]
            nbrs = data2[['category', 'Mean nbrs.', 'Median nbrs.','St.dev.', 'Min nbrs.', 'Max nbrs.']]

            print("========== Distance kDS metrics ==========")
            print(distances.to_latex(index = False))


            # Combute neighbors for DOUBLE range

            # Instantiate sets of global data collectors
            neighbors = []
            locs = []


            # For each set in the mins collection, compute values:
            counter = 0
            for s in sets:

                to_iterate = [n[0] for n in enumerate(G.nodes) if n[0] not in s]

                if counter < 4:
                    A = scipy.sparse.load_npz('data/Reachability_03_20kwh_UNDIRECTED.npz')
                if counter >= 4:
                    A = scipy.sparse.load_npz('data/Reachability_03_30kwh_UNDIRECTED.npz')
                if counter >= 8:
                    A = scipy.sparse.load_npz('data/Reachability_03_40kwh_UNDIRECTED.npz')

                # Statistics of sets:
                n_neighbors = []
                coords = []


                # For each row, compute:
                for i in to_iterate:
                    a = A[i,list(s)]
                    a_data = a.data

                    # Store neighbor a_data
                    n = len(a_data)
                    n_neighbors.append(n)

                    # Store coordinates
                    x = G.nodes[i]
                    y = G.nodes[i]
                    coords.append((x,y))

                
                # Append to full collection
                neighbors.append(n_neighbors)
                counter += 1
                locs.append(coords)

            data['category'] = list(zip(data.range, data.k))
            data2 = pd.DataFrame(data['category'].drop_duplicates())

            # Map n neighbors to each category
            data2['Mean nbrs.'] = [round(np.mean(i),3) for i in neighbors]
            data2['Median nbrs.'] = [int(np.median(i)) for i in neighbors]
            data2['St.dev.'] = [np.round(np.std(i),3) for i in neighbors]
            data2['Max nbrs.'] = [np.max(i) for i in neighbors]
            data2['Min nbrs.'] = [np.min(i) for i in neighbors]

            nbrs = data2[['category', 'Mean nbrs.', 'Median nbrs.','St.dev.', 'Min nbrs.', 'Max nbrs.']]  

            print("========== Neighbor kDS metrics ==========")
            print(nbrs.to_latex(index = False))