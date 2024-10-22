"""
QM9 (Quantum Mechanics 9)

A collection of molecules with up to nine heavy atoms (C, O, N, S)
used as a benchmark dataset for molecular property prediction and
graph-classification tasks.

This file is a loader for variations of the dataset.

"""
from typing import Optional

import torch
from pointgroup import PointGroup

from torch_geometric.datasets import QM9
import torch_geometric.transforms as T

from torch_canon.pointcloud import CanonEn as Canon

point_groups = {'C1':0, 'C1h':1, 'C1v':2, 'C2':3, 'C2d':4, 'C2h':5, 'C2v':6, 'C3':7, 'C3h':8, 'C3v':9, 'C4':10, 'Cs':11, 'Cinfv':12, 'Ci':13, 'Dinfh':14, 'D2':15, 'D2d':16, 'D2h':17, 'D3':18, 'D3d':19, 'D3h':20, 'D6h':21, 'Oh':22, 'Td':23, 'S2':24, 'S4':25}

targets = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0',
                      'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']

atomic_number_to_symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
atomic_symbol_to_number = {val:key for key,val in atomic_number_to_symbol.items()}

portions = {
    'fixed':(100_000,1_000),
    }

"""
Transformations 
~~~~~~~~~~~~~~~

"""

class Align(T.BaseTransform):
    def __init__(self, tol: Optional[float] = 1e-3):
        self.tol = tol

    def __call__(self, data):
        frame = Canon(tol=self.tol, save='all')
        align_pos, frame_R, frame_t = frame.get_frame(data.pos, data.z)
        data.align_pos = torch.from_numpy(align_pos)
        data.frame_R = frame_R
        data.frame_t = frame_t
        symmetric_elements = frame.symmetric_elements
        symmetric_edge_index = [[i,j] for symmetry_element in symmetric_elements for i in symmetry_element for j in symmetry_element]
        projection_edge_index = [[symmetry_element[0],symmetry_element[j]] for symmetry_element in symmetric_elements for j in range(1,len(symmetry_element))]
        data.project_edge_index = torch.tensor(projection_edge_index, dtype=torch.long).T
        data.symmetric_edge_index = torch.tensor(symmetric_edge_index, dtype=torch.long).T
        asu = frame.simple_asu
        asu_edge_index = [[i,j] for i in asu for j in asu]
        data.asu_edge_index = torch.tensor(asu_edge_index, dtype=torch.long).T
        return data

class AddPG(T.BaseTransform):
    def __call__(self, data):
        try:
            symbols = [atomic_number_to_symbol[atomic_num.item()] for atomic_num in data.z]
            pg = PointGroup(data.pos, symbols).get_point_group()
            data.pg = pg
        except:
            data.pg = 'C1'
        return data

"""
====
Main
====
"""
if __name__ == '__main__':
    pre_transform = []
    pre_transform.append(AddPG())
    pre_transform.append(Align(tol=0.01))

    transform = []

    pth = './data/qm9_align'

    dataset = QM9(root=pth, pre_transform=T.Compose(pre_transform), transform=T.Compose(transform))
    for data in dataset:
        print(data)
        exit()
