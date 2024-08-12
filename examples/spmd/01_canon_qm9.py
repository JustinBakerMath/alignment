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

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.stats import wasserstein_distance_nd

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from pointgroup import PointGroup

from torch_canon.E3Global.CategoricalPointCloud import CatFrame as Frame


# Setup
# -----
parser = argparse.ArgumentParser()
parser.add_argument('--n_data', type=int, default=100, help='Number of data.')
parser.add_argument('--n_g_act', type=int, default=100, help='Number of data transformations by group action.')
parser.add_argument('--frq_log', type=int, default=10, help='How often to log progress.')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

qm9 = QM9(root='./data/qm9-2.4.0/')
frame = Frame()

atomic_number_to_symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}

pg_map = ["C1","C1h","C1v","C2","C2h","C2v","C3","C3h","C3v","Cs","Cinfv","Ci","Dinfh","D2d","D2h","D3","D3d","D3h","D6h","Td","S2",]
pg_map = {pg: idx for idx, pg in enumerate(pg_map)}

pg_losses = [0 for _ in pg_map]
pg_counts = [0 for _ in pg_map]
data_rank1_loss, data_rank1_count = 0,0
data_rank2_loss, data_rank2_count = 0,0
data_rank3_loss, data_rank3_count = 0,0

# Helper Functions
# ----------------
def addCounter(counter1, counter2, datatype):
    for item in counter2:
        if item not in counter1:
            counter1[item] = counter2[item]
        else:
            counter1[item] += counter2[item]
    return counter1

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
#MPI.Init_thread(MPI.THREAD_MULTIPLE)
rank = comm.Get_rank()
size = comm.Get_size()

n_data = len(qm9[:args.n_data])
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
    symbols = [atomic_number_to_symbol[z] for z in cat_data]
        
    try:
        pg = PointGroup(normalized_data, symbols).get_point_group()
    except:
        pg = 'C1'

    local_losses = []
    for i in range( args.n_g_act):
        loss = compute_loss(i, pc_data, normalized_data, cat_data)

        if data_rank == 1:
            data_rank1_loss += loss
            data_rank1_count += 1
        elif data_rank == 2:
            data_rank2_loss += loss
            data_rank2_count += 1
        else:
            data_rank3_loss += loss
            data_rank3_count += 1

        pg_losses[pg_map[pg]] += loss
        pg_counts[pg_map[pg]] += 1
            

pg_loss_total = comm.reduce(np.array(pg_losses), op=MPI.SUM, root=0)
pg_count_total = comm.reduce(np.array(pg_counts), op=MPI.SUM, root=0)
rank1_loss_total = comm.reduce(data_rank1_loss, op=MPI.SUM, root=0)
rank1_count_total = comm.reduce(data_rank1_count, op=MPI.SUM, root=0)
rank2_loss_total = comm.reduce(data_rank2_loss, op=MPI.SUM, root=0)
rank2_count_total = comm.reduce(data_rank2_count, op=MPI.SUM, root=0)
rank3_loss_total = comm.reduce(data_rank3_loss, op=MPI.SUM, root=0)
rank3_count_total = comm.reduce(data_rank3_count, op=MPI.SUM, root=0)

# Results
# -------
if rank == 0:

    print(f'Rank 1 Count: {rank1_count_total},',
          f'Rank 2 Count: {rank2_count_total},',
          f'Rank 3 Count: {rank3_count_total}')

    print(f'Rank 1 Loss: {rank1_loss_total/(rank1_count_total+1e-16):.5f},',
          f' Rank 2 Loss: {rank2_loss_total/(rank2_count_total+1e-16):.5f},',
          f' Rank 3 Loss: {rank3_loss_total/(rank3_count_total+1e-16):.5f}')

    for idx, val in enumerate(pg_loss_total):
        count = pg_count_total[idx]
        pg = list(pg_map.keys())[idx]
        if count == 0:
            continue
        print(f'Point Group {list(pg_map.keys())[idx]} : ({count}, {val/count:.5f})')

# MPI Finalize
MPI.Finalize()

#print(f'Rank 1 Loss: {data_rank1_loss/(data_rank1_count+1e-16):.5f},',
      #f' Rank 2 Loss: {data_rank2_loss/(data_rank2_count+1e-16):.5f},',
      #f' Rank 3 Loss: {data_rank3_loss/(data_rank3_count+1e-16):.5f}')
#
#for key, dct in pg_losses.items():
    #val = dct['loss']/dct['count']
    #print(f'\tPoint Group {key} : {val}')
