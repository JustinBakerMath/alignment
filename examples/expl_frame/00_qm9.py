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
frame = Frame(tol=1e-2, save='all')

atomic_number_to_symbol = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'
    }
loss = 0
recon_loss = 0

# Helper Functions
# ----------------
def compute_loss(data, data_transformed):
    loss = wasserstein_distance_nd(data, data_transformed)
    return loss


np.random.seed(42)


# Main Loop
# ---------
for idx,data in enumerate(qm9[:args.n_data]):

    logging.info(f"Completed {idx+1}/{args.n_data} iterations.")

    pc_data = data.pos
    cat_data = data.z.numpy()
    normalized_data, frame_R, frame_t = frame.get_frame(pc_data, cat_data)

    smiles = ''.join([atomic_number_to_symbol[cat] for cat in cat_data])

    loss += compute_loss(pc_data, normalized_data)

    #inv_R = torch.tensor(R.from_matrix(frame_R).inv().as_matrix(), dtype=torch.float32)
    inv_R = torch.linalg.inv(frame_R)
    recon_data = (inv_R @ normalized_data.T).T + frame_t
    r_loss = compute_loss(pc_data, recon_data)
    if r_loss > 1e-4:
        logging.info(f'Loss {smiles}: {r_loss:.4f}')
        break
    recon_loss += r_loss

logging.info(f'Average move {loss/args.n_data:.4f}')
logging.info(f'Reconstruction loss {recon_loss/args.n_data:.4f}')
logging.info(f'Time: {time()-start_time:.4f}')
logging.info('Done!')
