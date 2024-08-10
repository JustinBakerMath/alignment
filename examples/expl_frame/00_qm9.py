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

from time import time

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.stats import wasserstein_distance_nd

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from pointgroup import PointGroup

sys.path.append('./torch_canon/')
from E3Global.CategoricalPointCloud import CatFrame as Frame


# Setup
# -----
parser = argparse.ArgumentParser()
parser.add_argument('--n_data', type=int, default=100, help='Random seed')
parser.add_argument('--frq_log', type=int, default=10, help='Random seed')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
start_time = time()

qm9 = QM9(root='./data/qm9-2.4.0/')
frame = Frame()

atomic_number_to_symbol = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'
    }
loss = 0

# Helper Functions
# ----------------
def compute_loss(i, pc_data, normalized_data, cat_data):
    random_rotation = R.random().as_matrix()
    random_translation = np.random.rand(3)

    g_pc_data = (random_rotation @ (pc_data + random_translation).numpy().T).T
    g_normalized_data, rot = frame.get_frame(g_pc_data, cat_data)
    loss = wasserstein_distance_nd(normalized_data, g_normalized_data)
    return loss


np.random.seed(42)


# Main Loop
# ---------
for idx,data in enumerate(qm9[:args.n_data]):

    logging.info(f"Completed {idx+1}/{args.n_data} iterations.")

    pc_data = data.pos
    cat_data = data.z.numpy()
    smiles = ''.join([atomic_number_to_symbol[cat] for cat in cat_data])
    print(smiles)

    data_rank = torch.linalg.matrix_rank(pc_data)
    normalized_data, rot = frame.get_frame(pc_data, cat_data)

    loss += compute_loss(idx, pc_data, normalized_data, cat_data)

logging.info(f'Average move {loss/args.n_data:.4f}')
logging.info(f'Time: {time()-start_time:.4f}')
logging.info('Done!')
