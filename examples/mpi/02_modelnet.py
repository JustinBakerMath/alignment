'''
Parallel Alignment of ModelNet40
===============================
This script is used to normalize the ModelNet40 dataset using the CategoricalPointCloud class.
In addition, it performs a random rotation and translation of the point cloud and calculates the Wasserstein distance between the original and the transformed point cloud.
It does so in parallel for all the molecules in the ModelNet40 dataset.
'''

# Start up
# -------
import argparse
import random
import logging
import sys
from tqdm import tqdm

from mpi4py import MPI

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.stats import wasserstein_distance_nd

import torch

sys.path.append('./datasets/')
from modelnet40 import ModelNet40Dataset

from torch_canon.pointcloud import CanonEn as Canon


# Setup
# -----
parser = argparse.ArgumentParser()
parser.add_argument('--n_data', type=int, default=100, help='Number of data.')
parser.add_argument('--n_g_act', type=int, default=2, help='Number of data transformations by group action.')
parser.add_argument('--frq_log', type=int, default=10, help='How often to log progress.')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

modelnet40 = ModelNet40Dataset(root='./data/modelnet/')
frame = Canon(tol=1e-2)

atomic_number_to_symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}

loss, count = 0,0
loss_total = 0
count_total = 0



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

    g_pc_data = (random_rotation @ (pc_data + random_translation).T).T
    g_normalized_data, rot = frame.get_frame(g_pc_data, cat_data)
    loss = wasserstein_distance_nd(normalized_data, g_normalized_data)
    return loss


# MPI Setup
# ---------
comm = MPI.COMM_WORLD
#MPI.Init_thread(MPI.THREAD_MULTIPLE)
rank = comm.Get_rank()
size = comm.Get_size()


# MAIN RANK
# ---------
if rank == 0:
    # Example data
    n_data = len(modelnet40[:args.n_data])
    batch_size = 1
    print('Batch size:', batch_size)
    task_index = 0
    tasks_sent = 0
    seed = 42

    # Distribute initial tasks
    for i in range(1, size):
        print(f"Main sending task to worker {i}")
        if task_index < n_data:
            batch_end = min(task_index + batch_size, n_data)
            for j in range(task_index, batch_end):
                comm.send(modelnet40[j], dest=i, tag=1)
            task_index += batch_size
            tasks_sent += 1

        print(f"Task index: {task_index}")

    with tqdm(total=n_data) as pbar:
        # Receive results and send new tasks dynamically
        while tasks_sent > 0:
            result_dict = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            tasks_sent -= 1

            loss_total += result_dict['loss']
            count_total += result_dict['count']

            if task_index < n_data:
                batch_end = min(task_index + batch_size, n_data)
                comm.send(modelnet40[task_index:batch_end], dest=result_dict['worker'], tag=1)
                task_index += batch_size
                tasks_sent += 1

            pbar.update(task_index - pbar.n)

        for i in range(1, size):
            comm.send(None, dest=i, tag=0)

else:
    # WORKER LOOP
    #------------
    while True:
        data = comm.recv(source=0, tag=MPI.ANY_TAG)
        if data is None:
            break

        # Main Loop
        # ---------
        pc_data = data.pos
        cat_data = data.x
        #print(f"Worker {rank} - {idx}: {smiles}")

        normalized_data, rot = frame.get_frame(pc_data, cat_data)
        
        local_losses = []
        for i in range( args.n_g_act):
            loss = compute_loss(i, pc_data, normalized_data, cat_data)

            loss += loss
            count += 1
    
        comm.send({'loss': loss, 'count': count}, dest=0, tag=1)
        print(f"Worker {rank} - {idx}: {loss:.5f}")

# MPI Finalize
MPI.Finalize()

# Results
# -------
if rank == 0:

    print(f'Rank 1 Count: {count_total},')

    print(f'Rank 1 Loss: {loss_total/(count_total+1e-16):.5f},')
