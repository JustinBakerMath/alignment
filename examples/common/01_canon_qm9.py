'''
Parallel Alignment of QM9
=========================
This script is used to normalize the QM9 dataset using the CategoricalPointCloud class.
In addition, it performs a random rotation and translation of the point cloud and calculates the Wasserstein distance between the original and the transformed point cloud.
It does so in parallel for all the molecules in the QM9 dataset.
'''

# Start up
# -------
import argparse
import sys
import logging

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.stats import wasserstein_distance_nd

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from pointgroup import PointGroup

from torch_canon.pointcloud import CanonEn as Canon


# Setup
# -----
parser = argparse.ArgumentParser()
parser.add_argument('--n_data', type=int, default=1, help='Random seed')
parser.add_argument('--n_g_act', type=int, default=1, help='Random seed')
parser.add_argument('--tol', type=float, default=1e-3, help='Tolerance for frame computation.')
parser.add_argument('--err', type=float, default=1e-6, help='Level of noise added to the point cloud.')
parser.add_argument('--frq_log', type=int, default=10, help='Random seed')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

qm9 = QM9(root='./data/qm9-2.4.0/')
frame = Canon(tol=args.tol)

atomic_number_to_symbol = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'
    }
pg_losses = {}
data_rank1_loss, data_rank1_count, data_rank1_moved = 0,0,0
data_rank2_loss, data_rank2_count, data_rank2_moved = 0,0,0
data_rank3_loss, data_rank3_count, data_rank3_moved = 0,0,0

# Helper Functions
# ----------------
def compute_loss(i, pc_data, normalized_data, cat_data):
    random_rotation = R.random().as_matrix()
    random_translation = np.random.rand(3)
    random_err = np.random.rand(*pc_data.shape) * args.err

    g_pc_data = (random_rotation @ (pc_data + random_translation + random_err).numpy().T).T
    shuffle = np.random.permutation(len(g_pc_data))
    g_pc_data = g_pc_data[shuffle]
    g_cat_data = cat_data[shuffle]
    g_normalized_data, g_frame_R, g_frame_t = frame.get_frame(g_pc_data, g_cat_data)
    loss = wasserstein_distance_nd(normalized_data, g_normalized_data)
    return loss


np.random.seed(42)


# Main Loop
# ---------
for idx,data in enumerate(qm9[:args.n_data]):
    # generate smiles using atomic_number_to_symbol and data.z
    smiles = ''.join([atomic_number_to_symbol[z] for z in data.z.numpy()])


    if idx % args.frq_log == 0:
        logging.info(f"Process Completed {idx+1}/{args.n_data} iterations.")

    pc_data = data.pos
    cat_data = data.z.numpy()
    data_rank = torch.linalg.matrix_rank(pc_data)

    normalized_data, frame_R, frame_t = frame.get_frame(pc_data, cat_data)
    symbols = [atomic_number_to_symbol[z] for z in cat_data]

    moved = compute_loss(0, pc_data, normalized_data, cat_data)
    if data_rank == 1:
        data_rank1_moved += moved
    elif data_rank == 2:
        data_rank2_moved += moved
    else:
        data_rank3_moved += moved
        
    for i in range(args.n_g_act):
        loss = compute_loss(i, pc_data, normalized_data, cat_data)
        if loss > 1e-2:
            try:
                pg = PointGroup(normalized_data, symbols).get_point_group()
            except:
                pg = 'C1'
            logging.info(f'{idx}: ({smiles}, {pg}) - {loss:.5f}')

        if data_rank == 1:
            data_rank1_loss += loss
            data_rank1_count += 1
        elif data_rank == 2:
            data_rank2_loss += loss
            data_rank2_count += 1
        else:
            data_rank3_loss += loss
            data_rank3_count += 1
            

# Results
# -------
print(f'Data Rank 1 Loss: {data_rank1_loss/(data_rank1_count+1e-16):.5f},',
      f' Data Rank 2 Loss: {data_rank2_loss/(data_rank2_count+1e-16):.5f},',
      f' Data Rank 3 Loss: {data_rank3_loss/(data_rank3_count+1e-16):.5f}')

print(f'Data Rank 1 Count: {data_rank1_count},',
      f'Data Rank 2 Count: {data_rank2_count},',
      f'Data Rank 3 Count: {data_rank3_count}')

print(f'Data Rank 1 Moved: {data_rank1_moved},',
      f' Data Rank 2 Moved: {data_rank2_moved},',
      f' Data Rank 3 Moved: {data_rank3_moved}')
