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

from mpi4py import MPI
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


# MPI Setup
# ---------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n_data = len(qm9[:args.n_data])
n_g_actions = 5
chunk_size = n_data // size
start_idx = rank * chunk_size
end_idx = (rank + 1) * chunk_size if rank != size - 1 else n_data

if rank == 0:
    seed = 42
else:
    seed = None
seed = comm.bcast(seed, root=0)
np.random.seed(seed + rank)


# Main Loop
# ---------
for idx,data in enumerate(qm9[start_idx:end_idx]):

    if rank==0 and (idx+1) % args.frq_log == 0:  
        logging.info(f"Process {rank}: Completed {idx+1}/{chunk_size} iterations.")

    pc_data = data.pos
    cat_data = data.z.numpy()

    data_rank = torch.linalg.matrix_rank(pc_data)
    normalized_data, rot = frame.get_frame(pc_data, cat_data)

    loss += compute_loss(idx, pc_data, normalized_data, cat_data)

loss_total = comm.reduce(loss, op=MPI.SUM, root=0)

# MPI Finalize
comm.Barrier()
MPI.Finalize()

if rank == 0:
    logging.info(f'Average move {loss_total/n_data:.4f}')
    logging.info(f'Time: {time()-start_time:.4f}')
    logging.info('Done!')