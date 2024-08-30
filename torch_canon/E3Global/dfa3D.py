import ast
from torch_canon.utilities import get_key

def construct_dfa(encoding, graph):
    dfa_set = list()
    for edge in graph:
        value = str([encoding[edge[0]], encoding[edge[1]]])
        dfa_set.append(value)
    return dfa_set

def convert_partition(hopcroft, dist_hash, g_hash, r_encoding, g_encoding):
    hashed_edges = list(tuple(ast.literal_eval(k)) for k in hopcroft._partition.keys())
    ret_edges = []
    ret_graph = []
    for hashed_edge in hashed_edges:
        # This is the actual geometry
        left_node_hash, right_node_hash = hashed_edge
        left_radii = get_key(dist_hash, left_node_hash[0])
        left_angles = get_key(g_hash, left_node_hash[1])
        right_radii = get_key(dist_hash, right_node_hash[0])
        right_angles = get_key(g_hash, right_node_hash[1])

        left_radii = [item for sublist in left_radii for tuple_item in sublist for item in tuple_item]
        left_angles = [item for sublist in left_angles for tuple_item in sublist for item in tuple_item]
        right_radii = [item for sublist in right_radii for tuple_item in sublist for item in tuple_item]
        right_angles = [item for sublist in right_angles for tuple_item in sublist for item in tuple_item]

        # These are the actual nodes
        test_left_g = get_key(g_encoding, left_node_hash[1])
        test_left_r = get_key(r_encoding, left_node_hash[0])
        left_node_idxs = [i for i in test_left_g if i in test_left_r]

        test_right_g = get_key(g_encoding, right_node_hash[1])
        test_right_r = get_key(r_encoding, right_node_hash[0])
        right_node_idxs = [i for i in test_right_g if i in test_right_r]

        if left_radii<right_radii:
          ret_edges.append(left_radii+left_angles+right_radii+right_angles)
          ret_graph.append([left_node_idxs,right_node_idxs])
        else:
          ret_edges.append(right_radii+right_angles+left_radii+left_angles)
          ret_graph.append([right_node_idxs,left_node_idxs])

    indexed_edges = sorted(enumerate(ret_edges), key=lambda x: x[1])
    sorted_inidces = [i for i,_ in indexed_edges]
    ret_graph = [ret_graph[i] for i in sorted_inidces]
    return ret_graph
