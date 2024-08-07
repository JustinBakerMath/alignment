'''

Notation:
    ch: Convex Hull
    us: Unit Sphere
    pc: Point Cloud
    s3: S3 Group
    t: Translation Group
'''

from abc import ABCMeta
import ast

import torch
import numpy as np

from torch_canon.utilities import *
from torch_canon.Hopcroft import PartitionRefinement

from torch_canon.E3Global.align3D import align_pc_t, align_pc_s3
from torch_canon.E3Global.encode3D import enc_us_catpc, enc_ch_pc
from torch_canon.E3Global.qhull import get_ch_graph

class Molecule:
    def __init__(self, data=None, cat_data=None):
        self.pos = data
        self.z = cat_data

class CatFrame(metaclass=ABCMeta):
    def __init__(self, tol=1e-2, *args, **kwargs):
        super().__init__()
        self.tol = tol

    def traverse(self, sorted_graph, us_data, us_rank):
        edge = 0
        v0 = sorted_graph[edge][0][0]
        if us_rank == 1:
            return [v0]
        s0 = us_data[v0]

        v1 = None
        while v1 is None and edge < len(sorted_graph):
            possible_indices = sorted_graph[edge][1]
            possible_indices = [i for i in possible_indices if i != v0]
            for idx in possible_indices:
                if np.abs(np.dot(s0, us_data[idx])) > self.tol:
                    v1 = idx
                    break
            if v1 is None:
                edge += 1

        if us_rank == 2:
            return [v0, v1]

        v2 = self.v2_subroutine(v0, v1, edge, sorted_graph, us_data, us_rank)
        if v2 is None:
            v2 = self.v2_subroutine(v1, v0, edge, sorted_graph, us_data, us_rank)
        
        assert v2 is not None, 'v2 is None'

        return [v0, v1, v2]

    def v2_subroutine(self, v0, v1, edge, sorted_graph, us_data, us_rank):
        s0 = us_data[v0]
        s1 = us_data[v1]
        v2 = None
        while v2 is None and edge < len(sorted_graph):
            if v1 in sorted_graph[edge][0]:
                possible_indices = sorted_graph[edge][1]
                possible_indices = [i for i in possible_indices if i != v0]
                possible_indices = [i for i in possible_indices if i != v1]
                for idx in possible_indices:
                    cond1 = np.abs(np.dot(s0, us_data[idx])) > self.tol
                    cond2 = np.abs(np.dot(s1, us_data[idx])) > self.tol
                    if cond1 and cond2:
                        v2 = idx
                        break
            if v2 is None:
                edge += 1
        return v2


    def convert_partition(self, dist_hash, g_hash, r_encoding, g_encoding):
        edges = list(tuple(ast.literal_eval(k)) for k in self.hopcroft._partition.keys())
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
        return sorted(ret_edges), ret_graph


    def construct_dfa(self, encoding, graph):
        dfa_encoding = {}
        dfa_set = list()
        for edge in graph:
            value = str([encoding[edge[0]], encoding[edge[1]]])
            dfa_encoding[(edge[0], edge[1])] = value
            dfa_set.append(value)
        return dfa_set, dfa_encoding


    def get_frame(self, data, cat_data, *args, **kwargs):
        data = check_type(data) # Assert Type

        # TRANSLATION GROUP
        # -----------------
        data, frame_t = align_pc_t(data) # Translation group alignment
        cntr_data = data.clone() # TODO: Don't copy just use indexing
        indices = torch.linalg.norm(data, axis=1) > self.tol
        data, cat_data = data[indices], cat_data[indices]

        # ROTATION GROUP
        # --------------
        # Unit Sphere Encoding
        dist_hash, r_encoding, us_data = enc_us_catpc(data, cat_data)
        
        # Build Convex Hull Graph
        us_rank = torch.linalg.matrix_rank(us_data, tol=self.tol)
        us_n = us_data.shape[0]
        ch_graph = get_ch_graph(us_data, us_rank, us_n)

        bool_lst = [i in ch_graph for i in range(us_n)]
        assert all(bool_lst), 'Convex Hull is not correct'

        # GET GEOMETRIC ENCODING
        adj_list = build_adjacency_list(ch_graph)
        dg = direct_graph(ch_graph)
        g_hash, g_encoding = enc_ch_pc(us_data, adj_list, us_rank)

        # COMBINE ENCODINGS
        n_encoding = {}
        # for each node combine ENCODINGS
        for i in range(us_n):
            n_encoding[i] = (r_encoding[i], g_encoding[i])

        # CONSTRUCT DFA
        dfa, edge_encoding = self.construct_dfa(n_encoding, dg)
        self.hopcroft = PartitionRefinement(dfa)
        out = self.hopcroft.refine(dfa)
        k = list(self.hopcroft._partition.keys())[0]
        for value in self.hopcroft._partition[k]:
            e = [edge for edge, enc in edge_encoding.items() if enc == value]

        sorted_edges, sorted_graph = self.convert_partition(dist_hash, g_hash, r_encoding, g_encoding)
        pth = self.traverse(sorted_graph, us_data, us_rank)
        return align_pc_s3(cntr_data, us_data, pth)
