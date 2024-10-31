'''
PyTest for torch_canon/pointcloud/canon.py
==========================================
'''

import pytest
import torch
import math

from torch_canon.pointcloud.qhull import get_ch_graph
from torch_canon.utilities import build_adjacency_list

from torch_canon.pointcloud.canon import (enc_us_pc, enc_us_catpc, enc_ch_pc)


# Predefined Objects
#-------------------
oneEQone = [[1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0+1e-16, 0.0, 0.0],
            ]
twoEQtwo = [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [2.0, 0.0, 0.0],
             [0.0, 2.0, 0.0],
             [0.0, 1.0+1e-16, 0.0],
             [1.0+1e-16, 0.0, 0.0],
             ]
threeEQthree = [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0],
             [0.0, 1.0+1e-16, 0.0],
             [1.0+1e-16, 0.0, 0.0],
             [0.0, 0.0, 1.0+1e-16],
             [2.0, 0.0, 0.0],
             [0.0, 2.0, 0.0],
             [0.0, 0.0, 2.0],
             ]
fourEQfour = [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0],
             [1.0, 1.0, 0.0],
             [1.0+1e-16, 0.0, 0.0],
             [0.0, 0.0, 1.0+1e-16],
             [2.0, 0.0, 0.0],
             [0.0, 2.0, 0.0],
             [0.0, 0.0, 2.0],
             ]

oneTOone = [[1],[1,1],[1,1,1]]
twoTOone = [[2],[1,2],[1,1,2]]
threeTOone = [[2],[1,2],[1,3,2]]

oneTOtwo = [[1 for _ in range(i+1)] for i in range(1,len(twoEQtwo))]


# Unit Tests
#-----------
def unittest_enc_us_pc(val,tol):
    '''
    The important parts of the encoding are that
     1. points are projected onto the unit sphere
     2. locally close points are pooled together
     3. distances are sorted
     4. correct return values
    '''
    data_map = {1:oneEQone, 2:twoEQtwo, 3:threeEQthree, 4:fourEQfour}
    data = torch.tensor(data_map[val], dtype=torch.float32)
    for incr in range(val,len(data)+1):
        dists_hash, encoding, arr = enc_us_pc(data[:incr], tol=tol)
        # 1. Check
        assert arr.norm(dim=1).allclose(torch.ones(len(arr)))
        # 2. Check
        assert pytest.approx(arr, abs=tol) == data[:val]/torch.linalg.norm(data[:val], axis=1).reshape(-1,1)
        # 3. Check
        for key in dists_hash.keys():
            assert sorted(key) == list(key)
        # 4. Check
        assert len(dists_hash) == val

def unittest_enc_us_catpc(val,tol):
    '''
    The important parts of the encoding are that
     1. points are projected onto the unit sphere
     2. locally close points are pooled together
     3. distances are sorted
    '''
    data_map = {1:oneEQone, 2:twoEQtwo, 3:threeEQthree, 4:fourEQfour}
    cat_map = {1:[oneTOone, oneTOtwo],
               2:[twoTOone],
               3:[threeTOone]}
    cat_data = cat_map[val]
    for idx, cat in enumerate(cat_data):
        data = torch.tensor(data_map[idx+1], dtype=torch.float32)
        for i,incr in enumerate(range(idx+1,len(data)+1)):
            cat_i = torch.tensor(cat[i], dtype=torch.float32)
            dists_hash, encoding, arr, sorted_local_mask, local_mask = enc_us_catpc(data[:incr], cat_i, tol=tol)
            # 1. Check
            assert arr.norm(dim=1).allclose(torch.ones(len(arr)))
            # 2. Check
            assert pytest.approx(arr, abs=tol) == data[:idx+1]/torch.linalg.norm(data[:idx+1], axis=1).reshape(-1,1)
            # 3. Check
            for key in dists_hash.keys():
                assert sorted(key) == list(key)


def unittest_enc_ch_pc(val, tol):
    '''
    The important parts of the encoding are that
     1. distances are encoded correctly
     2. angles are ordered
    '''
    data = torch.tensor(threeEQthree[:val], dtype=torch.float32)
    rank = torch.linalg.matrix_rank(data)
    ch_graph = get_ch_graph(data, rank, val)
    adj_list = build_adjacency_list(ch_graph)
    g_hash, encoding = enc_ch_pc(data, adj_list, rank, tol=tol)
    # 1. Check
    for key in g_hash.keys():
        angles = []
        for angle, dist in key:
            angles.append(angle)
            assert dist == pytest.approx(math.pi/2)
        # 2. Check
        assert sorted(angles) == angles
    pass



# Pytests
#--------
@pytest.mark.parametrize('val',[1, 2, 3, 4])
@pytest.mark.parametrize('tol',[1e-12])
def test_enc_us_pc(val, tol):
    unittest_enc_us_pc(val, tol)

@pytest.mark.parametrize('val',[1,2,3])
@pytest.mark.parametrize('tol',[1e-12])
def test_enc_us_catpc(val, tol):
    unittest_enc_us_catpc(val, tol)

@pytest.mark.parametrize('val',[1,2,3])
@pytest.mark.parametrize('tol',[1e-12])
def test_enc_ch_pc(val, tol):
    unittest_enc_ch_pc(val, tol)
