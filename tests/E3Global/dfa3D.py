
'''
DFA 3D
======

Includes:
    - Unit Sphere (US)
    - Convex Hull (CH)

'''

import pytest
import ast
import torch
import math

from torch_canon.utilities import build_adjacency_list
from torch_canon.E3Global.dfa3D import construct_dfa, convert_partition
from torch_canon.E3Global.encode3D import enc_us_pc, enc_us_catpc, enc_ch_pc
from torch_canon.Hopcroft import PartitionRefinement

encodings = [[('dhash0', 'ghash0'), ('dhash0', 'ghash0'), ('dhash0', 'ghash0'), ('dhash0','ghash0')],
             [('dhash0', 'ghash0'), ('dhash1', 'ghash1'), ('dhash0', 'ghash0'), ('dhash1','ghash1')],
             [('dhash0', 'ghash0'), ('dhash1', 'ghash1'), ('dhash2', 'ghash2'), ('dhash3','ghash3')]]

graphs = [[[0, 1], [1, 2], [2, 3], [3, 0]],
            [[0, 1], [1, 2], [2, 0], [0, 2]],
            [[0, 1], [1, 2], [2, 3], [3, 0]]]

'''
 [[[1], [4]], [[1], [5]], [[1], [6, 8]], [[1], [10, 11]], [[1], [9]], [[1], [0]], [[3], [2]], [[3], [4]], [[3], [5]], [[3], [6, 8]], [[3], [6, 8]], [[3], [7]], [[3], [10, 11]], [[3], [10, 11]], [[3], [9]], [[2], [3]], [[2], [6, 8]], [[2], [7]], [[2], [0]], [[4], [1]], [[4], [3]], [[4], [6, 8]], [[4], [9]], [[5], [1]], [[5], [3]], [[5], [10, 11]], [[5], [10, 11]], [[5], [0]], [[6, 8], [1]], [[6, 8], [3]], [[6, 8], [4]], [[6, 8], [7]], [[6, 8], [0]], [[6, 8], [3]], [[6, 8], [2]], [[6, 8], [10, 11]], [[6, 8], [0]], [[7], [3]], [[7], [2]], [[7], [6, 8]], [[7], [0]], [[10, 11], [3]], [[10, 11], [5]], [[10, 11], [6, 8]], [[10, 11], [0]], [[10, 11], [1]], [[10, 11], [3]], [[10, 11], [5]], [[10, 11], [9]], [[9], [1]], [[9], [3]], [[9], [4]], [[9], [10, 11]], [[0], [1]], [[0], [2]], [[0], [5]], [[0], [6, 8]], [[0], [6, 8]], [[0], [7]], [[0], [10, 11]]]
'''

dist_hash = {'d_a':'dhash0', 'd_b':'dhash1', 'd_c':'dhash2', 'd_d':'dhash3'}
g_hash = {'g_a':'ghash0', 'g_b':'ghash1', 'g_c':'ghash2', 'g_d':'ghash3'}
r_encoding = {0:'dhash0', 1:'dhash1', 2:'dhash2', 3:'dhash3'}
g_encoding = {0:'ghash0', 1:'ghash1', 2:'ghash2', 3:'ghash3'}

unit_cube = [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0],
             [1.0, 1.0, 0.0],
             [1.0, 0.0, 1.0],
             [0.0, 1.0, 1.0],
             [1.0, 1.0, 1.0],
             [-1.0, 0.0, 0.0],
             [0.0, -1.0, 0.0],
             [0.0, 0.0, -1.0],
             [-1.0, -1.0, 0.0],
             [-1.0, 0.0, -1.0],
             [0.0, -1.0, -1.0],
             [-1.0, -1.0, -1.0],
             ]



# Unit Tests
#-----------
def unittest_construct_dfa(encoding, graph):
    dfa_set = construct_dfa(encoding, graph)
    assert len(dfa_set) == len(graph)


def unittest_convert_partition(encoding, graph):
    dfa = construct_dfa(encoding, graph)
    hopcroft = PartitionRefinement(dfa)
    hopcroft.refine(dfa)
    edges = list(tuple(ast.literal_eval(k)) for k in hopcroft._partition.keys())
    sorted_graph = convert_partition(hopcroft, dist_hash, g_hash, r_encoding, g_encoding)
    print(sorted_graph)
    print('---'*20)
    pass



        
# Pytests
#--------
@pytest.mark.parametrize('encoding',encodings)
@pytest.mark.parametrize('graph',graphs)
def test_construct_dfa(encoding, graph):
    unittest_construct_dfa(encoding, graph)

@pytest.mark.parametrize('graph',graphs)
@pytest.mark.parametrize('encoding',encodings)
def test_convert_partition(encoding, graph):
    unittest_convert_partition(encoding, graph)
