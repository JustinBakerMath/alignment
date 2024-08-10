import ast
from torch_canon.utilities import get_key

def construct_dfa(encoding, graph):
    dfa_set = list()
    for edge in graph:
        value = str([encoding[edge[0]], encoding[edge[1]]])
        dfa_set.append(value)
    return dfa_set

def convert_partition(hopcroft, dist_hash, g_hash, r_encoding, g_encoding):
    edges = list(tuple(ast.literal_eval(k)) for k in hopcroft._partition.keys())
    ret_edges = []
    ret_graph = []
    for edge in edges:
        a,b = edge
        r0 = get_key(dist_hash, a[0])
        g0 = get_key(g_hash, a[1])
        r1 = get_key(dist_hash, b[0])
        g1 = get_key(g_hash, b[1])
        ret_edges.append([(r0,g0),(r1,g1)])

        r0 = get_key(r_encoding, a[0])
        r1 = get_key(r_encoding, b[0])
        ret_graph.append([r0,r1])

    indexed_edges = sorted(enumerate(ret_edges), key=lambda x: x[1])
    sorted_inidces = [i for i,_ in indexed_edges]
    ret_edges = [element for index, element in indexed_edges]
    ret_graph = [ret_graph[i] for i in sorted_inidces]
    return ret_graph
