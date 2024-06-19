"""Defining file for the automatic conduit sizing of APDUDS

This script requires that `swmm_api`, `pandas`, and `pathlib` be installed within the Python
environment you are running this script in.

This file contains the following major functions:

    * find_nodes_flooding - Finds which nodes are flooding based on the report
    * dict_edges_in_path - Selects and sorts edges in graph based on current diameter and presence in paths of flooding nodes
    * change_diameter - Alter the inp-file and edge dictionary with the updated diameter of a conduit
    * attr_sizing - Main loop for the conduit sizing
"""

import pandas as pd
from pathlib import Path
from swmm_api.input_file import SwmmInput
from swmm_api import swmm5_run, SwmmReport


def find_nodes_flooding(rpt) -> list[int]:
    # extract summary of flooding nodes
    result = rpt.node_flooding_summary

    # filter nodes which have significant amount of flooding
    nodes = result.index[result["Total_Flood_Volume_10^6 ltr"] > 0.000].values.tolist()

    # modify such that only the node id is stored
    indices = [int(node[2:]) for node in nodes]

    return indices

def dict_edges_in_path(flooding_nodes: list[int], diam_list: list[float], nodes: pd.DataFrame, edges: pd.DataFrame) -> list[tuple[str, int]]:
    edge_dict = {}
    for edge_index, edge in edges.iterrows():

        # determine amount of bigger sizes than current diameter
        spots_left = len(diam_list) - 1 - diam_list.index(edge.diameter)
        factor = 20 * spots_left

        for node_id in flooding_nodes:
            edge_name = f"c_{edge_index}"
            conduit_list = nodes.at[node_id, "conduit_list"]
            # check if edge appears in path of flooding node
            if edge_name in conduit_list:
                # keep score of how many times an edge exists in path
                if edge_name in edge_dict:
                    edge_dict[edge_name] += 1 * factor 
                else:
                    edge_dict[edge_name] = 1 * factor
    return sorted(edge_dict.items(), key=lambda x:x[1], reverse=True)


def change_diameter(inp, edges: pd.DataFrame, edge_dict: list[tuple[str, int]], diam_list: list[float]):
    #print(edge_dict)
    check_zero = 0
    for edge, count in edge_dict:
        check_zero += count
        # only widen conduit if there is room left
        if count != 0:
            diam = float(inp.XSECTIONS[edge].height)
            i = diam_list.index(diam)
            new_diam = diam_list[i+1] 

            # change diameter in inp-file
            inp.XSECTIONS[edge].height = new_diam

            # change diameter in edges dataframe
            edges.at[int(edge[2:]), "diameter"] = new_diam

    # check if all edges have factor 0
    all_zero = False
    if check_zero == 0:
        all_zero = True
    return inp, edges, all_zero


# main loop of automatic sizing
def attr_sizing(filename, nodes, edges, settings):
    print("Calculating optimal conduit sizes")
    pathinp = Path(__file__).parent.parent / f'{filename}.txt'
    pathrpt = Path(__file__).parent.parent / f'{filename}.rpt'

    # read and run swmm-file
    inp = SwmmInput.read_file(pathinp)
    swmm5_run(pathinp)
    rpt = SwmmReport(pathrpt)

    # get flooding nodes
    flooding_nodes = find_nodes_flooding(rpt)
    #print(flooding_nodes)

    # while there are still nodes flooding, 
    # widen conduits until there is either no more flooding or no more conduits to widen
    while len(flooding_nodes) > 0:
        edges_dict = dict_edges_in_path(flooding_nodes, settings["diam_list"], nodes, edges)
        inp, edges, all_zero = change_diameter(inp, edges, edges_dict[:25], settings["diam_list"])
        if all_zero:
            print(f"\nThere are no more relevant conduits to widen, while nodes {flooding_nodes} still flood. \
                  Run the system again with different settings")
            break
        inp.write_file(pathinp)
        swmm5_run(pathinp)
        rpt = SwmmReport(pathrpt)
        flooding_nodes = find_nodes_flooding(rpt)
        #print(flooding_nodes)
    print("\nConduit sizing complete")
    return edges

