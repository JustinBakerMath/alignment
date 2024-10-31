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
import io
import logging
import random
import sys
from tqdm import tqdm

from mpi4py import MPI

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.stats import wasserstein_distance_nd

import torch
from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader

from pointgroup import PointGroup

from torch_canon.pointcloud import CanonEn as Canon


# Setup
# -----
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='benzene', help='Dataset name.')
parser.add_argument('--n_data', type=int, default=100, help='Number of data.')
parser.add_argument('--n_batch', type=int, default=10, help='Number of data per batch.')
parser.add_argument('--tol', type=float, default=1e-1, help='Tolerance for frame computation.')
parser.add_argument('--err', type=float, default=1e-6, help='Level of noise added to the point cloud.')
parser.add_argument('--frq_log', type=int, default=10, help='How often to log progress.')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# MPI Setup
# ---------
comm = MPI.COMM_WORLD
#MPI.Init_thread(MPI.THREAD_MULTIPLE)
rank = comm.Get_rank()
size = comm.Get_size()


# MAIN RANK
# ---------
if rank == 0:
    md17 = MD17(root='./data/md17/',name=args.name,force_reload=True)

    # Example data
    n_data = len(md17[:args.n_data])
    batch_size = min(args.n_batch, int(args.n_data/size))  # Initial batch size
    print('Batch size:', batch_size)
    task_index = 0
    tasks_sent = 0
    seed = 42

    gathered_data = []

    # Distribute initial tasks
    for i in range(1, size):
        #print(f"Main sending task to worker {i}")
        if task_index < n_data:
            batch_end = min(task_index + batch_size, n_data)
            buffer = io.BytesIO()
            torch.save(md17[task_index:batch_end], buffer)
            serialized_data = buffer.getvalue()
            comm.send(serialized_data, dest=i, tag=1)
            task_index += batch_size
            tasks_sent += 1

        #print(f"Task index: {task_index}")

    with tqdm(total=n_data) as pbar:
        # Receive results and send new tasks dynamically
        while tasks_sent > 0:
            result_dict = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            tasks_sent -= 1

            data = result_dict['data']
            gathered_data = comm.gather(data, root=0)

            if task_index < n_data:
                batch_end = min(task_index + batch_size, n_data)
                #print(f"Main sending task to worker {result_dict['worker']}")
                buffer = io.BytesIO()
                torch.save(md17[task_index:batch_end], buffer)
                serialized_data = buffer.getvalue()
                comm.send(serialized_data, dest=i, tag=1)
                task_index += batch_size
                tasks_sent += 1

            #print(f"Task index: {task_index}")
            pbar.update(task_index - pbar.n)

        # Send stop signal to workers
        for i in range(1, size):
            comm.send(None, dest=i, tag=0)

else:
    # WORKER LOOP
    #------------
    while True:
        task = comm.recv(source=0, tag=MPI.ANY_TAG)
        if task is None:
            break

        buffer = io.BytesIO(task)
        task = torch.load(buffer)
        # Main Loop
        # ---------
        print(f"Worker {rank} received task with {len(task)} molecules.")
        for idx,data in enumerate(task):
            print('Worker', rank, 'processing data', idx)
            frame = Canon(tol=args.tol,save=True)
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
            
            #buffer = io.BytesIO()
            #torch.save(data, buffer)
            #serialized_data = buffer.getvalue()
            comm.send({'data':data, 'worker': rank}, dest=0, tag=rank)
            print('Worker', rank, 'sent data', idx)

# MPI Finalize
MPI.Finalize()

# Results
# -------
if rank == 0:

    for data in gathered_data:
        print(data)
        break
