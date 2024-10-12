'''
Parallel Alignment of MD17
=========================
This script is used to normalize the MD17 dataset using the CategoricalPointCloud class.
In addition, it performs a random rotation and translation of the point cloud and calculates the Wasserstein distance between the original and the transformed point cloud.
It does so in parallel for all the molecules in the MD17 dataset.
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
from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader

from pointgroup import PointGroup

sys.path.append('./torch_canon/')
from E3Global.CategoricalPointCloud import CatFrame as Frame


# Setup
# -----
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='benzene', help='Dataset name')
parser.add_argument('--n_data', type=int, default=100, help='Random seed')
parser.add_argument('--frq_log', type=int, default=10, help='Random seed')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
start_time = time()

md17 = MD17(root='./data/md17/', name=args.name)
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
for idx,data in enumerate(md17[:args.n_data]):

    logging.info(f"Completed {idx+1}/{args.n_data} iterations.")

    pc_data = data.pos
    cat_data = data.z.numpy()
    normalized_data, frame_R, frame_t = frame.get_frame(pc_data, cat_data)

logging.info(f'Time: {time()-start_time:.4f}')
logging.info('Done!')
