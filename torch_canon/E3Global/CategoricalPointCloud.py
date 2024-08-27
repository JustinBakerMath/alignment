'''
                         --variable shorthand--
Notation:
    ch: Convex Hull
    us: Unit Sphere
    pc: Point Cloud
    s3: S3 Group
    t: Translation Group
'''

from abc import ABCMeta

import torch
import numpy as np

from torch_canon.Hopcroft import PartitionRefinement
from torch_canon.utilities import build_adjacency_list, check_type, direct_graph

from torch_canon.E3Global.align3D import align_pc_t, align_pc_s3
from torch_canon.E3Global.dfa3D import construct_dfa, convert_partition
from torch_canon.E3Global.encode3D import enc_us_catpc, enc_ch_pc
from torch_canon.E3Global.geometry3D import check_colinear

from abc import ABCMeta

import numpy as np
from scipy.spatial import ConvexHull

def get_ch_graph(data, rank, n, *args, **kwargs):
    if n==1:
        return np.array([])
    elif n==2:
        return np.array([[0,1]])
    elif n==3:
        return np.array([[0,1],[1,2],[2,0]])
    else:
        hull =  ConvexHull(data, qhull_options='QJ')
        if hull.simplices[0].shape[0]==3:
            edges = set()
            for simplex in hull.simplices:
                for i in range(3):
                    edge = sorted([simplex[i], simplex[(i + 1) % 3]])
                    edges.add(tuple(edge))
            return np.array(list(edges))
        else:
            return hull.simplices


class CatFrame(metaclass=ABCMeta):
    def __init__(self, tol=1e-4, *args, **kwargs):
        super().__init__()
        self.tol = tol

    def get_frame(self, data, cat_data, *args, **kwargs):
        data = check_type(data) # Assert Type

        # TRANSLATION GROUP
        # -----------------
        data, frame_t = align_pc_t(data) # Translation group alignment
        cntr_data = data.clone() # TODO: Don't copy just use indexing
        dists = torch.linalg.norm(data, axis=1)
        indices = dists > self.tol
        data, cat_data = data[indices], cat_data[indices]

        # ROTATION GROUP
        # --------------
        # Unit Sphere Encoding
        dist_hash, r_encoding, us_data = enc_us_catpc(data, cat_data, tol=self.tol/dists.max()*dists.min())
        
        # Build Convex Hull Graph
        us_rank = torch.linalg.matrix_rank(us_data, tol=self.tol)
        us_n = us_data.shape[0]
        ch_graph = get_ch_graph(us_data, us_rank, us_n)

        bool_lst = [i in ch_graph for i in range(us_n)]
        assert all(bool_lst), 'Convex Hull is not correct'

        # Encode Convex Hull Geometry
        adj_list = build_adjacency_list(ch_graph)
        dg = direct_graph(ch_graph)
        g_hash, g_encoding = enc_ch_pc(us_data, adj_list, us_rank)

        # COMBINE ENCODINGS
        n_encoding = {}
        # for each node combine ENCODINGS
        for i in range(us_n):
            n_encoding[i] = (r_encoding[i], g_encoding[i])

        # CONSTRUCT DFA
        dfa = construct_dfa(n_encoding, dg)
        self.hopcroft = PartitionRefinement(dfa)
        self.hopcroft.refine(dfa)
        sorted_graph = convert_partition(self.hopcroft, dist_hash, g_hash, r_encoding, g_encoding)
        pth = self.traverse(sorted_graph, us_data, us_rank)
        data, frame_R = align_pc_s3(cntr_data, us_data, pth)
        return data, frame_R, frame_t

    def traverse(self, sorted_graph, us_data, us_rank):

        # ~~~
        # Start on the first set of symmetric edges and with the first node
        vert0_idx, sym_edge_idx = self.find_vert0(sorted_graph)
        if us_rank == 1:
            return [vert0_idx]
        vert0_vec = us_data[vert0_idx]

        # ~~~
        # Find the second node
        vert1_idx = self.find_vert1(vert0_idx, vert0_vec, sym_edge_idx, sorted_graph, us_data)
        if us_rank == 2:
            return [vert0_idx, vert1_idx]
        vert1_vec = us_data[vert1_idx]

        v2 = self.v2_subroutine(vert0_idx, vert1_idx, sym_edge_idx, sorted_graph, us_data, us_rank)
        if v2 is None:
            v2 = self.v2_subroutine(vert1_idx, vert0_idx, sym_edge_idx, sorted_graph, us_data, us_rank)
        
        assert v2 is not None, f'v2 is None\n {vert0_idx},{vert1_idx}\n \n {sorted_graph}'

        return [vert0_idx, vert1_idx, v2]

    def find_vert0(self, sorted_graph):
        symmetry_group = 0
        left_edge_vertices = 0
        first_vertex = 0
        return sorted_graph[symmetry_group][left_edge_vertices][first_vertex], symmetry_group

    def find_vert1(self, vert0_idx, vert0_vec, sym_edge_idx, sorted_graph, us_data):
        vert1_idx = None
        # First we will check among all right nodes connected to this node
        while vert1_idx is None and sym_edge_idx < len(sorted_graph):

            test_vert_idxs = sorted_graph[sym_edge_idx][1] # Get connected edges in symmetry edge
            test_vert_idxs = [i for i in test_vert_idxs if i != vert0_idx] # Ignore if it's v0 (may happen due to symmetry BD)

            # testing vertices
            for idx in test_vert_idxs:
                test_vert_vec = us_data[idx]
                
                # If they are co-linear then ignore
                colinear_test = check_colinear(vert0_vec, test_vert_vec, self.tol)
                if colinear_test > self.tol:
                    vert1_idx = idx
                    break
            
            # Try to find the next possible connected node
            if vert1_idx is None:
              sym_edge_idx += 1
              while (sym_edge_idx < len(sorted_graph)) and (not vert0_idx in sorted_graph[sym_edge_idx][0]):
                sym_edge_idx += 1
              

        if vert1_idx is not None:
            return vert1_idx

        # Now we need to check among all left nodes
        else:
            sym_edge_idx = 0
            while vert1_idx is None and sym_edge_idx < len(sorted_graph):
    
                test_vert_idxs = sorted_graph[sym_edge_idx][0] # Get connected edges in symmetry edge
                test_vert_idxs = [i for i in test_vert_idxs if i != vert0_idx] # Ignore if it's connected to v0
    
                # testing vertices
                for idx in test_vert_idxs:
                    test_vert_vec = us_data[idx]
                    
                    # If they are co-linear then ignore
                    colinear_test = np.abs(np.dot(vert0_vec, test_vert_vec)) 
                    if colinear_test > self.tol**2:
                        vert1_idx = idx
                        break
                
                # Now we can consider any node so consider any edge
                if vert1_idx is None:
                    sym_edge_idx += 1
                
            if vert1_idx is not None:
                  return vert1_idx
              # In some case where the only non co-linear point is connected to some point far far away
            else:
                sym_edge_idx = 0
                while vert1_idx is None and sym_edge_idx < len(sorted_graph):
        
                    test_vert_idxs = sorted_graph[sym_edge_idx][1] # Get connected edges in symmetry edge
                    test_vert_idxs = [i for i in test_vert_idxs if i != vert0_idx] # Ignore if it's connected to v0
        
                    # testing vertices
                    for idx in test_vert_idxs:
                        test_vert_vec = us_data[idx]
                        
                        # If they are co-linear then ignore
                        if np.abs(np.dot(vert0_vec, test_vert_vec)) > self.tol:
                            vert1_idx = idx
                            break
                    
                    # Now we can consider any node so consider any edge
                    if vert1_idx is None:
                        sym_edge_idx += 1
                return vert1_idx

    def v2_subroutine(self, vert0_idx, vert1_idx, sym_edge_idx, sorted_graph, us_data, us_rank):
        s0 = us_data[vert0_idx]
        s1 = us_data[vert1_idx]
        v2 = None
        while sym_edge_idx < len(sorted_graph) and v2 is None:
            if vert1_idx in sorted_graph[sym_edge_idx][0]:
                possible_indices = sorted_graph[sym_edge_idx][1]
                possible_indices = [i for i in possible_indices if i != vert0_idx]
                possible_indices = [i for i in possible_indices if i != vert1_idx]
                for idx in possible_indices:
                    cond1 = np.abs(np.dot(s0, us_data[idx])) > self.tol
                    cond2 = np.abs(np.dot(s1, us_data[idx])) > self.tol
                    if cond1 and cond2:
                        v2 = idx
                        break
            if v2 is None:
                sym_edge_idx += 1
        if v2 is None:
            sym_edge_idx = 0
            while sym_edge_idx < len(sorted_graph) and v2 is None:
                if vert0_idx in sorted_graph[sym_edge_idx][0]:
                    possible_indices = sorted_graph[sym_edge_idx][1]
                    possible_indices = [i for i in possible_indices if i != vert0_idx]
                    possible_indices = [i for i in possible_indices if i != vert1_idx]
                    for idx in possible_indices:
                        cond1 = np.abs(np.dot(s0, us_data[idx])) > self.tol
                        cond2 = np.abs(np.dot(s1, us_data[idx])) > self.tol
                        if cond1 and cond2:
                            v2 = idx
                            break
                if v2 is None:
                    sym_edge_idx += 1
        if v2 is None:
            sym_edge_idx = 0
            while sym_edge_idx < len(sorted_graph) and v2 is None:
                if not(vert0_idx in sorted_graph[sym_edge_idx][0]) and not (vert1_idx in sorted_graph[sym_edge_idx][0]):
                    possible_indices = sorted_graph[sym_edge_idx][0]
                    possible_indices = [i for i in possible_indices if i != vert0_idx]
                    possible_indices = [i for i in possible_indices if i != vert1_idx]
                    for idx in possible_indices:
                        cond1 = np.abs(np.dot(s0, us_data[idx])) > self.tol
                        cond2 = np.abs(np.dot(s1, us_data[idx])) > self.tol
                        if cond1 and cond2:
                            v2 = idx
                    sym_edge_idx += 1
                else:
                    v2 = sorted_graph[sym_edge_idx][0][0]
        return v2
