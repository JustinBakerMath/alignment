'''
PyTest for torch_canon/pointcloud/
==========================================

'''

import pytest
import logging

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.stats import wasserstein_distance_nd
from pointgroup import PointGroup

from torch_geometric.datasets import QM9

from torch_canon.pointcloud import CanonEn as Canon

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')

# Predefined Objects
#-------------------
qm9_test_idx = [20,27]
atomic_number_to_symbol = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'
    }

def compute_loss(frame, pc_data, cat_data, normalized_data):
    random_rotation = R.random().as_matrix()
    random_translation = np.random.rand(3)

    g_pc_data = (random_rotation @ (pc_data + random_translation).numpy().T).T
    g_normalized_data, g_frame_R, g_frame_t = frame.get_frame(g_pc_data, cat_data)
    loss = wasserstein_distance_nd(normalized_data, g_normalized_data)
    return loss

# Unit Tests
#-----------
def unittest_qm9(indices, tol):
  qm9 = QM9(root='./data/qm9-2.4.0/')
  test_data = []
  for i, data in enumerate(qm9):
    if i in indices:
      try:
        pg = PointGroup(normalized_data, symbols).get_point_group()
      except:
        pg = 'C1'
      smiles = ''.join([atomic_number_to_symbol[z] for z in data.z.numpy()])
      test_data.append([data, smiles, pg])
      print(f'{i}: ({smiles}, {pg})')
    if i>max(indices):
      break

  frame = Canon(tol=tol)
  for i, (data, smiles, pg) in enumerate(test_data):
    pc_data = data.pos
    cat_data = data.z
    normalized_data, frame_Q, frame_t = frame.get_frame(pc_data, cat_data)
    for _ in range(16):
        loss = compute_loss(frame,pc_data, cat_data, normalized_data)
        assert loss < tol, f'{i} ({smiles}, {pg}): {loss}'
  

# Pytests
#--------
@pytest.mark.parametrize('tol',[1e-1, 1e-2])
@pytest.mark.parametrize('indices',[qm9_test_idx])
def test_qm9(indices, tol):
    unittest_qm9(indices, tol)
