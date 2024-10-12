import ast
import torch
import numpy as np
from torch_canon.utilities import get_key
from .complete import check_colinear
from scipy.spatial.transform import Rotation

def construct_dfa(encoding, graph):
    dfa_set = list()
    for edge in graph:
        value = str([encoding[edge[0]], encoding[edge[1]]])
        dfa_set.append(value)
    return dfa_set

def convert_partition(hopcroft, dist_hash, g_hash, r_encoding, g_encoding, zero_mask):
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

    aligned_graph = ret_graph.copy()
    zero_elems = []
    for index_idx, index_bool in enumerate(zero_mask):
        if not index_bool:
            zero_elems.append(index_idx)
            for sym_idx, symmetry_group in enumerate(aligned_graph):
                sources, targets = symmetry_group
                sources = [source + 1 if source >= index_idx else source for source in sources]
                targets = [target + 1 if target >= index_idx else target for target in targets]
                aligned_graph[sym_idx] = [sources, targets]

    if len(zero_elems) > 0:
        aligned_graph.append([zero_elems, zero_elems])

    return ret_graph, aligned_graph


def traversal(sorted_graph, us_adj_dict, us_data, us_rank, zero_mask):
    symmetry_group = 0
    path = []
    visited_symmetry_groups = [False for _ in range(len(sorted_graph))]

    path.append(sorted_graph[symmetry_group][0][0])
    while False in visited_symmetry_groups and len(path) < us_data.shape[0]:

        k = 0
        while sorted_graph[symmetry_group][1][k] in path and k < len(sorted_graph[symmetry_group][1]) - 1:
            k += 1
        target = sorted_graph[symmetry_group][1][k]

        if target in path:
            visited_symmetry_groups[symmetry_group] = True
            symmetry_group = (symmetry_group + 1)%len(sorted_graph)
            continue
        else:
            path.append(target)
            for idx, (sources, _) in enumerate(sorted_graph):
                if target in sources and not visited_symmetry_groups[idx]:
                    symmetry_group = idx
                    break

    
    # need to align with the boolean mask of zero_mask -- everytime the mask is false we need to bump the path index by 1
    aligned_path = path.copy()
    for index_idx, index_bool in enumerate(zero_mask):
        if not index_bool:
            for path_idx, path_val in enumerate(aligned_path):
                if path_val >= index_idx:
                    aligned_path[path_idx] = path_val + 1

    return path, aligned_path



def construct_symmetries(data, symmetric_elements, tol):
    rank = np.linalg.matrix_rank(data,tol=tol)
    #print('RANK:', rank)
    idx_near_zero = list(np.where(np.linalg.norm(data, axis=1) <tol)[0])
    symmetric_elements = [sym for sym in symmetric_elements if not set(idx_near_zero).intersection(set(sym))]

    set_lengths = [len(sym) for sym in symmetric_elements]
    min_set_length = min(set_lengths)
    max_set_length = max(set_lengths)
    I = np.eye(data.shape[1])

    # if the count of set lenghts ==1 is larger than the rank then C1
    sigma = True
    if rank == 1: #(C1, Cinfv, Dinfh) (E, i)
        if sum([x==1 for x in set_lengths]) >  3:
            #print('DETECTED C1')
            return [I] # C1, (E)
        elif min_set_length == 1:
            #print('DETECTED Cinfv')
            return [I] # Cinfv, (E)
        else:
            #print('DETECTED Dinfh')
            R = np.eye(data.shape[1])
            R[2][2] = -1
            #assert np.linalg.norm(data+data@R.T) < tol
            return [I, R] # Dinfh, (E, i)
    elif rank == 2:
        if len(symmetric_elements) == 1:
            ##print('DETECTED C2')
            # get R as the 
            r = len(symmetric_elements[0])
            i=1
            while np.cross(data[0], data[i]).sum() < tol:
                i += 1
            n = np.cross(data[0], data[i])
            n = n/np.linalg.norm(n)
            Rs = []
            for idx in range(r):
                # Construct rotation matrix
                R = Rotation.from_rotvec((2*np.pi)*(idx+1)/r * n).as_matrix()
                Rs.append(R)
            Ps = []
            if r%2==0:
                # get the edge-edge and node-node reflections
                for i in range(0,r-2): # I think we are missing one reflection
                    v = data[i]
                    v = v/np.linalg.norm(v)
                    n = np.cross(n, v)
                    P = np.eye(data.shape[1]) - 2*np.outer(n, n)
                    diff = data + data@P.T
                    projection = np.dot(diff, n)
                    #assert np.linalg.norm(projection) < tol
                    Ps.append(P)
                    v = (data[i] + data[i+1])/2
                    if np.linalg.norm(v) > tol:
                        v = v/np.linalg.norm(v)
                        n = np.cross(n, v)
                        P = np.eye(data.shape[1]) - 2*np.outer(n, n)
                        diff = data + data@P.T
                        projection = np.dot(diff, n)
                        #assert np.linalg.norm(projection) < tol
            else:
                # get the edge-face reflections
                pass
            return [I] + Rs + Ps

        elif sum([x==1 for x in set_lengths]) >  3:
            #print('DETECTED C1')
            return [np.eye(data.shape[1])]
        elif min_set_length==max_set_length:
            #print('DETECTED C2h')
            n = np.cross(data[0], data[1])
            elems = [sym[0] for sym in symmetric_elements]
            #print(symmetric_elements)
            Rs = []
            for v in data[elems]:
                n = n/np.linalg.norm(n)
                v = v/np.linalg.norm(v)
                n = np.cross(n, v)
                R = np.eye(data.shape[1]) - 2*np.outer(n, n)
                diff = data + data@R.T
                projection = np.dot(diff, n)
                #print(projection)
                #assert np.linalg.norm(projection) < tol
                Rs.append(R)
            return [I] + Rs
        elif min_set_length == 1:
            test_set = [sym[0] for sym in symmetric_elements if len(sym) == 1]
            # check if they are all colinear
            test = test_set[0]
            for idx in range(1,len(test_set)):
                test_data = torch.tensor(data[test])
                test_data_set = torch.tensor(data[test_set[idx]])
                if not check_colinear(test_data, test_data_set, tol):
                    sigma = False
                    break
            if sigma and len(test_set) > 1:
                #print('DETECTED C2h')
                n = np.cross(data[0], data[1])
                Rs = []
                for v in data[test_set]:
                    n = n/np.linalg.norm(n)
                    v = v/np.linalg.norm(v)
                    n = np.cross(n, v)
                    R = np.eye(data.shape[1]) - 2*np.outer(n, n)
                    diff = data + data@R.T
                    projection = np.dot(diff, n)
                    #assert np.linalg.norm(projection) < tol
                    Rs.append(R)
                return [I] + Rs
            else:
                #print('DETECTED C2v')
                n = np.cross(data[0], data[1])
                v = data[test]
                n = n/np.linalg.norm(n)
                v = v/np.linalg.norm(v)
                n = np.cross(n, v)
                R = np.eye(data.shape[1]) - 2*np.outer(n, n)
                diff = data + data@R.T
                projection = np.dot(diff, n)
                #assert np.linalg.norm(projection) < tol
                return [np.eye(data.shape[1]), R]


    return [np.eye(data.shape[1])]
