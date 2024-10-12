'''
PyTest for torch_canon/pointcloud/qhull.py
==========================================

'''

import pytest
import torch
import math

from torch_canon.pointcloud.qhull import get_ch_graph_2d

# Predefined Objects
#-------------------
tetrahedron = [[1.0, 1.0, 1.0],
               [-1.0, -1.0, 1.0],
               [-1.0, 1.0, -1.0],
               [1.0, -1.0, -1.0]]

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
def unittest_cartesian_to_spherical(direction, tol
        ):
  sqrt2_2 = math.sqrt(2)/2
  dir2fn = {'x': cartesian2spherical_xtheta,
                   'y': cartesian2spherical_ytheta,
                   'z': cartesian2spherical_ztheta}
  dir2idx = {'x': 0, 'y': 1, 'z': 2}
  fn = dir2fn[direction]

  # axial test
  #~~~~~~~~~~
  pos_cart = torch.zeros(3)
  pos_cart[dir2idx[direction]] = 1.0
  pos_sph = fn(*pos_cart)
  assert pos_sph == pytest.approx((1.0, 0, 0), rel=tol, abs=tol)

  # negative axial test
  #~~~~~~~~~~~~~~~~~~~~
  pos_cart = torch.zeros(3)
  pos_cart[dir2idx[direction]] = -1.0
  pos_sph = fn(*pos_cart)
  assert pos_sph == pytest.approx((1.0, torch.pi, 0), rel=1e-7, abs=1e-7)

  # planar test
  #~~~~~~~~~~~~
  pos_cart = torch.zeros(3)
  idx = [i for i in range(3) if i != dir2idx[direction]]
  pos_cart[idx] = sqrt2_2
  pos_sph = fn(*pos_cart)
  assert pos_sph == pytest.approx((1.0, torch.pi/2, torch.pi/4), rel=1e-7, abs=1e-7)

  # rotate planar test
  #~~~~~~~~~~~~~~~~~~~~~
  pos_cart = torch.zeros(3)
  idx = [i for i in range(3) if i != dir2idx[direction]]
  pos_cart[idx[0]] = sqrt2_2
  pos_cart[idx[1]] = -sqrt2_2
  pos_sph = fn(*pos_cart)
  assert pos_sph == pytest.approx((1.0, torch.pi/2, 3*torch.pi/4), rel=1e-7, abs=1e-7)

  # negative planar test
  #~~~~~~~~~~~~~~~~~~~~~
  pos_cart = torch.zeros(3)
  idx = [i for i in range(3) if i != dir2idx[direction]]
  pos_cart[idx] = -sqrt2_2
  pos_sph = fn(*pos_cart)
  assert pos_sph == pytest.approx((1.0, torch.pi/2, -3*torch.pi/4), rel=1e-7, abs=1e-7)

  # rotate planar test
  #~~~~~~~~~~~~~~~~~~~~~
  pos_cart = torch.zeros(3)
  idx = [i for i in range(3) if i != dir2idx[direction]]
  pos_cart[idx[0]] = -sqrt2_2
  pos_cart[idx[1]] = sqrt2_2
  pos_sph = fn(*pos_cart)
  assert pos_sph == pytest.approx((1.0, torch.pi/2, -torch.pi/4), rel=1e-7, abs=1e-7)


def unittest_planar_alignment(direction, positions, tol):
  dir2fn = {'xy': xy_planar_alignment,
              'xz': xz_planar_alignment,
              'zy': zy_planar_alignment}
  dir2idx = {'xy': 2, 'xz': 1, 'zy': 0}
  dir2axs = {'xy': 0,
             'xz': 0,
             'zy': 2}

  positions = torch.tensor(positions)
  aligned_positions, Q = dir2fn[direction](positions.unsqueeze(0), positions)

  idx = dir2idx[direction]
  for pos in aligned_positions:
    # test dynamic vector
    #~~~~~~~~~~~~~~~~~~~~
    assert pytest.approx(pos[idx], abs=tol, rel=tol) == 0.0

    # test static vector
    #~~~~~~~~~~~~~~~~~~~
    axs = dir2axs[direction]
    assert pytest.approx(pos[axs], abs=tol, rel=tol) == positions[axs]

def unittest_z_axis_alignment(vector, tol):
    pos = torch.tensor(vector, dtype=torch.float32)
    aligned_pos, Q = z_axis_alignment(pos.unsqueeze(0), pos)
    assert aligned_pos[0][0] == pytest.approx(0.0, abs=tol, rel=tol)
    assert aligned_pos[0][1] == pytest.approx(0.0, abs=tol, rel=tol)
    assert aligned_pos[0][2] == pytest.approx(pos.norm(), abs=tol, rel=tol)


# Pytests
#--------
@pytest.mark.parametrize('tol',[1e-12, 1e-16])
@pytest.mark.parametrize('direction',["x", "y", "z"])
def test_cartesian_to_spherical(direction, tol):
    unittest_cartesian_to_spherical(direction, tol)

@pytest.mark.parametrize('tol',[1e-6])
@pytest.mark.parametrize('pos', unit_cube)
@pytest.mark.parametrize('direction',["xy", "xz", "zy"])
def test_planar_align(direction, pos, tol):
    unittest_planar_alignment(direction, pos, tol)

@pytest.mark.parametrize('tol',[1e-6])
@pytest.mark.parametrize('pos', unit_cube)
def test_zaxis_align(pos, tol):
    unittest_z_axis_alignment(pos, tol)


if __name__=='__main__':
    '''
    tet = torch.tensor(tetrahedron, dtype=torch.float32)
    print(tet)
    unit_tet = tet/tet.norm(dim=0)
    edges = get_ch_graph_2d(unit_tet, 3, unit_tet.shape[0])
    print(edges)
    '''

    import sys
    from tqdm import tqdm

    sys.path.append('./datasets/')
    from modelnet40 import ModelNet40Dataset

    print('Loading ModelNet40 Dataset')
    train_ds = ModelNet40Dataset(root='./data/modelnet')
    print(train_ds)

    for i in tqdm(range(1)):
        print(train_ds[i].pos)
        pos = torch.tensor(train_ds[i].pos, dtype=torch.float32)
        get_ch_graph_2d(pos, 3, pos.shape[0])
