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
from torch_canon.E3Global.qhull import get_ch_graph

from abc import ABCMeta

import numpy as np
from scipy.spatial import ConvexHull

class PermCatFrame(metaclass=ABCMeta):
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
        dists = torch.linalg.norm(data, axis=1)

        # ROTATION GROUP
        # --------------
        # Unit Sphere Encoding
        dist_hash, r_encoding, us_data = enc_us_catpc(data, cat_data, tol=self.tol)

        # Build Convex Hull Graph
        us_rank = torch.linalg.matrix_rank(us_data, tol=self.tol)
        us_n = us_data.shape[0]
        ch_graph = get_ch_graph(us_data, us_rank, us_n)

        bool_lst = [i in ch_graph for i in range(us_n)]
        assert all(bool_lst), 'Convex Hull is not correct'

        # Encode Convex Hull Geometry
        us_adj_dict = build_adjacency_list(ch_graph)
        dg = direct_graph(ch_graph)
        g_hash, g_encoding = enc_ch_pc(us_data, us_adj_dict, us_rank, tol=self.tol)

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
        pth = self.traverse(sorted_graph, us_adj_dict, us_data, us_rank)
        data, frame_R = align_pc_s3(cntr_data, us_data, pth)
        frame_P = self.sort(sorted_graph)
        self._save(data, frame_P, frame_R, frame_t, sorted_graph)
        return data[frame_P], frame_P, frame_R, frame_t

    def traverse(self, sorted_graph, us_adj_dict, us_data, us_rank):

        # ~~~
        # Start on the first set of symmetric edges and with the first node
        vert0_idx, dfa_node_idx = self.find_vert0(us_adj_dict, sorted_graph)
        if us_rank == 1:
            return [vert0_idx]
        vert0_vec = us_data[vert0_idx]

        # ~~~
        # Find the second node
        vert1_idx, dfa_node_idx = self.find_vert1(vert0_idx, vert0_vec, dfa_node_idx, us_adj_dict, sorted_graph, us_data)
        if us_rank == 2:
            return [vert0_idx, vert1_idx]
        vert1_vec = us_data[vert1_idx]

        v2 = self.v2_subroutine(vert0_idx, vert1_idx, dfa_node_idx, sorted_graph, us_adj_dict, us_data, us_rank)
        if v2 is None:
            v2 = self.v2_subroutine(vert1_idx, vert0_idx, dfa_node_idx, sorted_graph, us_adj_dict, us_data, us_rank)

        assert v2 is not None, f'v2 is None\n {vert0_idx},{vert1_idx}\n \n {sorted_graph}'

        return [vert0_idx, vert1_idx, v2]

    def find_vert0(self, us_adj_dict, sorted_graph):
        symmetry_group = 0
        left_edge_vertices = 0
        first_vertex = 0
        vert0_idx = sorted_graph[symmetry_group][left_edge_vertices][first_vertex]
        return vert0_idx, symmetry_group

    def find_vert1(self, vert0_idx, vert0_vec, dfa_node_idx, us_adj_dict, sorted_graph, us_data):
        return us_adj_dict[vert0_idx][0], None

    def v2_subroutine(self, vert0_idx, vert1_idx, dfa_node_idx, sorted_graph, us_adj_dict, us_data, us_rank):
        vert2_idx = None
        vert0_vec = us_data[vert0_idx]
        vert1_vec = us_data[vert1_idx]

        for idx in us_adj_dict[vert1_idx]:
          if idx != vert0_idx:
            return idx


        # Reset the dfa_node_idx v1 was received from a right hand node
        if dfa_node_idx is None:
          dfa_node_idx = 0
          for i in range(len(sorted_graph)):
            if vert1_idx in sorted_graph[i][0]:
              dfa_node_idx = i
              break

        # # Check DFA among right nodes (order 0)
        # ~~~
        while vert2_idx is None and dfa_node_idx < len(sorted_graph):

            test_vert_idxs = sorted_graph[dfa_node_idx][1] # Get connected edges in symmetry edge
            test_vert_idxs = [i for i in test_vert_idxs if (i != vert0_idx) and (i!=vert1_idx)] # Ignore if it's v0 (may happen due to symmetry BD)
            us_adj_idxs = us_adj_dict[vert1_idx]
            test_vert_idxs = [i for i in test_vert_idxs if i in us_adj_idxs] # Ignore if it's not in the ch_graph

            # testing vertices
            for idx in test_vert_idxs:
                test_vert_vec = us_data[idx]

                # ignore if co-linear
                colinear0_bool = check_colinear(vert0_vec, test_vert_vec, self.tol)
                colinear1_bool = check_colinear(vert1_vec, test_vert_vec, self.tol)
                if (not colinear0_bool) and (not colinear1_bool):
                    vert2_idx = idx
                    return vert2_idx

            # iterate dfa
            if vert2_idx is None:
              dfa_node_idx += 1
              while (dfa_node_idx < len(sorted_graph)) and (not vert1_idx in sorted_graph[dfa_node_idx][0]):
                dfa_node_idx += 1

        return vert2_idx

    def sort(self, sorted_graph):
        idx = []
        for lst in sorted_graph:
            for nodes in lst:
                for node in nodes:
                    if node not in idx:
                        idx.append(node)
        return idx

    def _save(self, data, frame_P, frame_R, frame_t, dfa):
        self.data = data
        self.frame_P = frame_P
        self.frame_R = frame_R
        self.frame_t = frame_t
        self.dfa = dfa
        pass
