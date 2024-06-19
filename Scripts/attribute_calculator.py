"""Defining file for all the attribute calculation functions of step 2 of APDUDS

This script requires that `networkx`, `pandas`, `freud` and `numpy` be installed within the Python
environment you are running this script in.

This file contains the following major functions:

    * overflow_location - Finds the node which is closest to water
    * voronoi_area - Calculates the catchment area for each node using voronoi
    * adjusted_area - Re-calculates the area based on elevation of nearby nodes
    * flow_and_depth - Determine the flow direction and set the node depth
    * flow_amount - Determine the amount of water flow through each conduit
    * diameter_calc - Determine the appropriate diameter for eac conduit
    * uphold_min_depth - Moves all installation levels of pipes to correct location
    * cleaner_and_trimmer - Remove intermediate information and precision from the data, calculate path to overflow for each node
    * pump_capacity - Calculates the needed capacity for a pump 
    * attribute_calculations - Runs the entire attribute calculation process
    * tester - Only used for testing purposes
"""

import osmnx as ox
import networkx as nx
import pandas as pd
from freud.box import Box
from freud.locality import Voronoi
import numpy as np
from osm_extractor import centralizer
from swmm_formater import swmm_file_creator
from autom_sizing import attr_sizing
import terminal

def overflow_location(nodes: pd.DataFrame, coords: list):
    """Determines the node which is closest to water

    Args:
        nodes (pd.DataFrame): The node data of a network
        coords (list[float]): Coordinates of the bounding box

    Returns:
        int: id of the node closest to water
    """
    print('Calculation the best location for an overflow')

    # import relevant objects from OpenStreetMap
    cf = '["natural"~"water"]'
    osm_map = ox.graph_from_bbox(*coords, truncate_by_edge=True, retain_all=True, custom_filter=cf)
    osm_projected = ox.project_graph(osm_map)

    # find distance to nearest edge/node in the graph of water objects for each node in the conduit network
    dist_edges = ox.distance.nearest_edges(osm_projected, nodes.x, nodes.y, return_dist=True)[1]
    dist_nodes = ox.distance.nearest_nodes(osm_projected, nodes.x, nodes.y, return_dist=True)[1]

    # determine the id of the node which is closest to water
    if min(dist_edges) < min(dist_nodes):
        id = np.argmin(dist_edges) 
    else:
        id = np.argmin(dist_nodes)

    return id

def voronoi_area(nodes: pd.DataFrame, edges: pd.DataFrame):
    """Calculates the catchment area for the nodes using voronoi

    Args:
        nodes (pd.DataFrame): The node data of a network

    Returns:
        tuple([pd.DataFrame, Freud.locality.Voronoi]): Node data with added subcatchment area
        values, and freud voronoi object
    """

    nodes = nodes.copy()

    box = Box(Lx=nodes.x.max() * 2, Ly=nodes.y.max() * 2, is2D=True)
    points = np.array([[nodes.x[i], nodes.y[i], 0] for i in range(len(nodes))])

    voro = Voronoi()
    voro.compute((box, points))

    nodes["area"] = voro.volumes

    return nodes, voro

def flow_and_depth(nodes: pd.DataFrame, edges: pd.DataFrame, settings:dict):
    """Determines the direction of flow of the water (using Dijkstra's algorithm) and
    needed installation depth of the nodes based on the given settings.

    Args:
        nodes (DataFrame): The node data of a network
        edges (DataFrame): The conduit data of a network
        settings (dict): Parameters for the network

    Returns:
        tuple[DataFrame, DataFrame]: Node data with added depth and path values,
        and conduit data with "from" and "to" columns corrected
    """

    nodes = nodes.copy()
    edges = edges.copy()

    nodes, edges, graph = intialize(nodes, edges, settings)
    nodes["path"] = None
    nodes["depth"] = nodes["elevation"] - settings["min_depth"]
    end_points = settings["outfalls"]
    nodes.loc[end_points, "considered"] = True
    # Create a set of all the "to" "from" combos of the conduits for later calculations
    edge_set = [set([edges["from"][i], edges["to"][i]]) for i in range(len(edges))]

    i = 1
    while not nodes["considered"].all():
        # Using the number of connections to sort them will make leaf nodes be considered first,
        # which has a larger change to include more nodes in one dijkstra run
        leaf_nodes = nodes.index[nodes.connections == i].tolist()

        for node in leaf_nodes:
            if not nodes.at[node, "considered"]:
                path = determine_path(graph, node, end_points)
                nodes = set_paths(nodes, path)
                nodes = set_depth(nodes, edges, path, settings["min_slope"], edge_set)

                nodes.loc[path, "considered"] = True
        i += 1

    if "max_slope" in settings:
        nodes = uphold_max_slope(nodes, edges, edge_set, settings["max_slope"])

    edges = reset_direction(nodes, edges)
    return nodes, edges


def intialize(nodes: pd.DataFrame, edges: pd.DataFrame, settings: dict):
    """Add the needed columns to the node and edge datasets to facilitate the operations
    of later functions. Also creates a networkx graph for the dijkstra calculations

    Args:
        nodes (DataFrame): The node data of a network
        edges (DataFrame): The conduit data of a network
        settings (dict): Parameters for the network

    Returns:
        tuple[DataFrame, DataFrame, Graph]: The node and edge datasets with the needed columns
        added, and a networkx graph of the network
    """
    nodes["considered"] = False
    nodes["role"] = "node"
    nodes["path_overflow"] = None
    nodes.at[6, 'path_overflow'] = [] # only for Tuindorp, node with no edges

    # Some more complex pandas operations are needed to get the connection numbers in a few lines
    ruined_edges = edges.copy()
    edges_melted = ruined_edges[["from", "to"]].melt(var_name='columns', value_name='index')
    edges_melted["index"] = edges_melted["index"].astype(int)
    nodes["connections"] = edges_melted["index"].value_counts().sort_index()

    graph = nx.Graph()
    graph.add_nodes_from(list(nodes.index.values))

    # Add weights to each conduit based on elevation change
    for _, edge in edges.iterrows():
        slope = (nodes.at[int(edge["from"]), "elevation"] - nodes.at[int(edge["to"]), "elevation"])  / edge["length"]
        if  slope >= 0:
             graph.add_edge(edge["from"], edge["to"], weight = 1 * abs(slope) * edge["length"])
        else:
            graph.add_edge(edge["from"], edge["to"], weight = 10 * abs(slope) * edge["length"] )
    return nodes, edges, graph


def determine_path(graph: nx.Graph, start: int, ends: list[int]):
    """Determines the shortest path from a certain point to another point on a networkx graph
    using Dijkstra's shortes path algorithm

    Args:
        graph (Graph): A NetworkX Graph object of the network
        start (int): The index of the starting node
        end (int): The index of the end node

    Returns:
        list[int]: The indicies of the nodes which the shortes path passes through
    """
    shortest_length = np.inf
    best_path = []

    for end_point in ends:
        length, path = nx.single_source_dijkstra(graph, start, target=end_point)

        if length < shortest_length:
            best_path = path
            shortest_length = length

    # Generator expression is needed to remove the .0 that is added by networkx' dijkstra
    return [int(x) for x in best_path]


def set_paths(nodes: pd.DataFrame, path: list):
    """Determine the path to the outfall for each node, and add this to the node data

    Args:
        nodes (DataFrame): The node data for a network
        path (list[int]): The indicies of the nodes which a path passes through

    Returns:
        DataFrame: Node data with the relevant path values updated
    """

    for i, node in enumerate(path):

        if not nodes.loc[node, "path"]:
            nodes.at[node, "path"] = path[i:]

    return nodes

def set_paths_overflow(nodes: pd.DataFrame, path: list):
    """Determine the path to the overflow for each node, and add this to the node data

    Args:
        nodes (DataFrame): The node data for a network
        path (list[int]): The indicies of the nodes which a path passes through

    Returns:
        DataFrame: Node data with the relevant path values updated
    """
    for i, node in enumerate(path):
        if not nodes.loc[node, "path_overflow"]:
            nodes.at[node, "path_overflow"] = path[i:]

    return nodes


def set_depth(nodes: pd.DataFrame, edges: pd.DataFrame,
              path: list, min_slope: float, edge_set: list[set[int]]):
    """Set the depth of the nodes along a certain route using the given minimum slope.

    Args:
        nodes (DataFrame): The node data for a network
        edges (DataFrame): The conduit data for a network
        path (list): All the indicies of the nodes which the path passes through
        min_slope (float): The value for the minimum slope [m/m]

    Returns:
        DataFrame: The node data with the relevant depth values updated
    """

    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i+1]

        from_depth = nodes.at[from_node, "depth"]
        # Use the edge set to get the conduit index
        length = edges.at[edge_set.index(set([from_node, to_node])), "length"]
        new_to_depth = from_depth - min_slope * length

        # Only update the depth if the new depth is deeper than the current depth
        if new_to_depth < nodes.at[to_node, "depth"]:
            nodes.at[to_node, "depth"] = new_to_depth

    return nodes

def uphold_max_slope(nodes: pd.DataFrame, edges: pd.DataFrame,\
                     edge_set: list[set[int]], max_slope: float):
    """Checks if the conduits uphold the max slope rule, and alters/lowers the relevant nodes
    when this isn't the case

    Args:
        nodes (DataFrame): The node data for a network
        edges (DataFrame): The conduit data for a network
        edge_set (list[set[int]]): A list of sets of all the "from" "to" node combos
        of the conduits
        max_slope (float): The value of the maximum slope [m/m]

    Returns:
        DataFrame: The node data with the depth value updated were needed
    """

    for _, node in nodes.iterrows():
        path = node.path

        for i in range(len(path)-1):
            lower_node = path[-1-i]
            higher_node = path[-2-i]
            length = edges.at[edge_set.index(set([lower_node, higher_node])), "length"]

            if abs(nodes.at[lower_node, "depth"] - nodes.at[higher_node, "depth"])\
                 / length > max_slope:
                nodes.at[higher_node, "depth"] = nodes.at[lower_node, "depth"] + length * max_slope
    return nodes


def reset_direction(nodes: pd.DataFrame, edges: pd.DataFrame):
    """Flips the "from" and "to" columns for all conduits where needed if depth is reversed

    Args:
        nodes (DataFrame): The node data for a network
        edges (DataFrame): The conduit data for a network

    Returns:
        DataFrame: Conduit data with the "from" "to" order flipped were needed
    """

    for i, edge in edges.iterrows():
        if nodes.at[edge["from"], "depth"] < nodes.at[edge["to"], "depth"]:
            edges.at[i, "from"], edges.at[i, "to"] = edge["to"], edge["from"]

    return edges

def adjusted_area(nodes: pd.DataFrame, edges: pd.DataFrame):
    """Re-calculate the areas of all nodes based on elevation of nearby nodes.

    Args:
        nodes (DataFrame): The node data for a network
        edges (DataFrame): The conduit data for a network
        settings (dict): Network parameters

    Returns:
        tuple[DataFrame, DataFrame]: Node and conduit data with the adjusted area
    """

    for i, _ in nodes.iterrows():
        length_elevation_above, length_above, length_elevation_below, length_below = 0, 0, 0, 0 
        for _, edge in edges[edges["from"] == i].iterrows():
            length = edge["length"]
            elevation  = nodes.at[int(edge["to"]), "elevation"]
            if elevation - nodes.at[i, "elevation"] > 0:
                length_elevation_above += length * elevation
                length_above += length
            else: 
                length_elevation_below += length * elevation
                length_below += length
        try:
            eq_nodes_above = length_elevation_above / length_above
        except ZeroDivisionError:
            eq_nodes_above = 0
        try:
            eq_nodes_below = length_elevation_below / length_below
        except ZeroDivisionError:
            eq_nodes_below = 0

        if nodes.at[i, "elevation"] != 0:
            factor = (np.exp((eq_nodes_above - eq_nodes_below) / nodes.at[i, "elevation"]))**0.25
        else:
            factor = (np.exp((eq_nodes_above - eq_nodes_below) / elevation))**0.25
        nodes.at[i, "area"] = nodes.at[i, "area"] * factor

    return nodes, edges


def flow_amount(nodes: pd.DataFrame, edges: pd.DataFrame, settings: dict):
    """Calculate the amount of flow through each conduit, and convert peak rain value to m/s.

    Args:
        nodes (DataFrame): The node data for a network
        edges (DataFrame): The conduit data for a network
        settings (dict): Network parameters

    Returns:
        tuple[DataFrame, DataFrame]: Node and conduit data with the inflow and flow
        values added
    """

    nodes = nodes.copy()
    edges = edges.copy()

    nodes["inflow"] = nodes["area"] * (settings["peak_rain"] / (3.6e6))\
         * (settings["perc_inp"] / 100)
    edges["flow"] = 0
    edge_set = [set([edges["from"][i], edges["to"][i]]) for i in range(len(edges))]

    for _, node in nodes.iterrows():
        path = node["path"]

        for j in range(len(path)-1):
            edge = set([path[j], path[j+1]])
            edges.at[edge_set.index(edge), "flow"] += node["inflow"]

    return nodes, edges


def diameter_calc(edges: pd.DataFrame, diam_list: list[float]):
    """Sets all diameters to the smallest available size, unless it has 0 flow. 
       Exact conduit sizes will be calculated later

    Args:
        edges (DataFrame): The conduit data for a network
        diam_list (list[float]): List of the different usable diameter sizes for the
        conduits [m]

    Returns:
        DataFrame: Conduit data with diameter values added
    """

    edges["diameter"] = None
    for i, edge in edges.iterrows():
        if edge["flow"] == 0:
            edges.at[i, "diameter"] = diam_list[0]
        else:
            edges.at[i, "diameter"] = diam_list[0]
    return edges


def pump_capacity(edges: pd.DataFrame, settings: dict, area: float):
    edges["volume"] = None
    for i, _ in edges.iterrows():
        edges.at[i, "volume"] = 0.25 * np.pi * (edges.at[i, "diameter"])**2 * edges.at[i, "length"]
    volume = edges.volume.sum()
    discharge = volume / 12
    settings["pump_capacity"] = discharge
    print(f'The needed pump capacity is {discharge:.2f} m^3/h, this is {discharge/area*1000:.2f} mm/h')
    return settings


def uphold_min_depth(nodes: pd.DataFrame, edges: pd.DataFrame, settings: dict):
    """Move all pipes lower so that they follow the set minimum depth.

    Args:
        nodes (DataFrame): The node data for a network
        edges (DataFrame): The conduit data for a network
        settings (dict): Network parameters

    Returns:
        tuple[DataFrame, DataFrame]: Node and conduit data with the updated node depth
    """
    
    for i, node in nodes.iterrows():
        try: 
            nodes.at[i, "install_depth"] = float(node["depth"] - edges["diameter"][edges["from"].values == i].values.max())
        except ValueError: #Raised if outflow or overflow node is reached
            nodes.at[i, "install_depth"] = float(node["depth"] - edges["diameter"][edges["to"].values == i].values.max())
            pass

    return nodes, edges


def cleaner_and_trimmer(nodes: pd.DataFrame, edges: pd.DataFrame, settings: dict):
    """Remove the columns from the node and conduit dataframes which were only needed for the
    attribute calculations. Also round off the calculated values to realistic presicions

    Args:
        nodes (DataFrame): The node data for a network
        edges (DataFrame): The conduit data for a network

    Returns:
        tuple[DataFrame, DataFrame]: Cleaned up nodes and conduit data
    """

    #nodes = nodes.drop(columns=["considered", "path", "connections"])
    nodes = nodes.drop(columns=["considered", "connections"])

    # Special condition if data was obtained from a csv (only for testing purposes)
    if "Unnamed: 0" in nodes.keys():
        nodes = nodes.drop(columns=["Unnamed: 0"])
        edges = edges.drop(columns=["Unnamed: 0"])

    # cm precision for x, y and depth
    # m^2 precision for area
    # L precision for inflow
    nodes.x = nodes.x.round(decimals=2)
    nodes.y = nodes.y.round(decimals=2)
    nodes.area = nodes.area.round(decimals=0)
    nodes.depth = nodes.depth.round(decimals=2)
    nodes.inflow = nodes.inflow.round(decimals=3)
    nodes.install_depth = nodes.install_depth.round(decimals=4)

    # cm precision for length
    # L precision for flow
    edges.length = edges.length.round(decimals=2)
    edges.flow = edges.flow.round(decimals=3)

    nodes, edges, graph = intialize(nodes, edges, settings)
    nodes["considered"] = False
    end_points = settings["overflows"] + [6]
    nodes.loc[end_points, "considered"] = True
    i = 1
    while not nodes["considered"].all():
        
        for node in range(len(nodes)-1):
            if not nodes.at[node, "considered"]:
                path_overflow = determine_path(graph, node, settings['overflows'])
                nodes = set_paths_overflow(nodes, path_overflow)
                nodes.loc[path_overflow, "considered"] = True
        i += 1
    nodes = nodes.drop(columns=["considered", "connections"])

    return nodes, edges


def add_outfalls(nodes: pd.DataFrame, edges: pd.DataFrame, settings: dict):
    """Add extra nodes for the selected outfall and overflow nodes. Connect them up with new
    conduits

    Args:
        nodes (DataFrame): The node data for a network
        edges (DataFrame): The conduit data for a network
        settings (dict): Parameters for the network

    Returns:
        tuple[DataFrame, DataFrame]: The node and conduit data with extra nodes and conduits
        for the outfalls and overflows
    """
    nodes2 = nodes.copy()
    edges2 = edges.copy()
    graph = intialize(nodes2, edges2, settings)[2]
    for outfall in settings["outfalls"]:
        new_index = len(nodes)
        settings["to_outfall"].append(new_index)
        nodes.loc[new_index] = [nodes.at[outfall, "x"] + 5,
                                nodes.at[outfall, "y"] + 5,
                                nodes.at[outfall, "elevation"],
                                0,
                                "outfall",
                                determine_path(graph, outfall, settings["overflows"]),
                                [],
                                nodes.at[outfall, "depth"],
                                0,
                                nodes.at[outfall, "install_depth"]]

        edges.loc[len(edges)] = [outfall,
                                 new_index,
                                 1,
                                 0,
                                 settings["diam_list"][-1]]
        
    
    for overflow in settings["overflows"]:
        new_index = len(nodes)
        settings["to_overflows"].append(new_index)
        nodes.loc[new_index] = [nodes.at[overflow, "x"] + 5,
                                nodes.at[overflow, "y"] + 5,
                                nodes.at[overflow, "elevation"],
                                0,
                                "overflow",
                                [],
                                determine_path(graph, overflow, settings["outfalls"]),
                                nodes.at[overflow, "depth"],
                                0,
                                nodes.at[overflow, "install_depth"]]

    return nodes, edges

def path_to_conduits(nodes: pd.DataFrame, edges: pd.DataFrame):
    """Converts all paths from a list of nodes to a list of edges 

    Args:
        nodes (DataFrame): The node data for a network
        edges (DataFrame): The conduit data for a network

    Returns:
        tuple[DataFrame, DataFrame]: The node and conduit data which the attribute values updated
    """
    nodes["conduit_list"] = None
    for node_index, node in nodes.iterrows():
        path = node["path_overflow"]
        conduit_list = []
        for i in range(len(path)-1):
            from_node = path[i]
            to_node = path[i+1]

            # search for the wanted conduit, also in opposite direction, and add to conduit list
            conduit = edges.index[(edges["from"] == from_node) & (edges["to"] == to_node)].to_list() 
            conduit += edges.index[(edges["from"] == to_node) & (edges["to"] == from_node)].to_list()
            if len(conduit) > 0:
                conduit_list.append("c_" + str(conduit[0]))

        # set conduit_list 
        nodes.at[node_index, "conduit_list"] = conduit_list
    return nodes, edges


def attribute_calculation(nodes: pd.DataFrame, edges: pd.DataFrame, settings: dict):
    """Does the complete attribute calculation step for a given network

    Args:
        nodes (DataFrame): The node data for a network
        edges (DataFrame): The conduit data for a network
        settings (dict): Parameters for the network

    Returns:
        tuple[DataFrame, DataFrame]: The node and conduit data with newly added and updated
        attribute values
    """
    nodes, edges = centralizer(nodes, edges)
    nodes, voro = voronoi_area(nodes, nodes)
    area = nodes.area.sum()

    nodes, edges = flow_and_depth(nodes, edges, settings)
    nodes, edges = adjusted_area(nodes, edges)
    nodes, edges = flow_amount(nodes, edges, settings)
    edges = diameter_calc(edges, settings["diam_list"])
    nodes, edges = uphold_min_depth(nodes, edges, settings)
    nodes, edges = cleaner_and_trimmer(nodes, edges, settings)
    settings["to_overflows"] = []
    settings["to_outfall"] = []
    nodes, edges = add_outfalls(nodes, edges, settings)
    nodes, edges = path_to_conduits(nodes, edges)
    settings = terminal.step_3_input(settings)
    swmm_file_creator(nodes, edges, voro, settings, pump=False)
    edges = attr_sizing(settings["filename"], nodes, edges, settings)
    settings = pump_capacity(edges, settings, area)
    # remove conduit to outfall to make room for pump-conduit
    edges = edges[edges['from'] != settings['outfalls'][0]]
    # remake swmm file with pump
    swmm_file_creator(nodes, edges, voro, settings, pump=True)
    return nodes, edges, voro

def tester():
    """Only used for testing purposes
    """
    print("attribute_calculator script has run")


if __name__ == "__main__":
    tester()
