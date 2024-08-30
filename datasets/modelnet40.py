#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys    
import trimesh
from tqdm import tqdm

import numpy as np

from torch_geometric.data import InMemoryDataset, download_url, Data, extract_zip
import torch

class ModelNet40Dataset(InMemoryDataset):
    def __init__(self, root, name='train', transform=None, pre_transform=None, pre_filter=None):
        super(ModelNet40Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        if name == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif name == 'test':
            self.data, self.slices = torch.load(self.processed_paths[1])
        else:
            raise ValueError('name should be either train or test')

    @property
    def raw_file_names(self):
        return ['ModelNet40']

    @property
    def processed_file_names(self):
        return ['modelnet40_train_data.pt', 'modelnet40_test_data.pt']


    def download(self):
        if not os.path.exists(self.processed_paths[0]):
            # Check if the raw data needs to be downloaded
            if not all(os.path.exists(os.path.join(self.raw_dir, name)) for name in self.raw_file_names):
                url = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
                path = download_url(url, self.raw_dir)
                extract_zip(path, self.raw_dir)
        pass


    def process(self):
        data_list = []
        modelnet40_dir = self.raw_paths[0]
        objects = os.listdir(modelnet40_dir)  # List all directories (objects)
        train_data_list = []
        test_data_list = []
        for obj in objects:
            training_dir_path = os.path.join(modelnet40_dir, obj, "train")  # Path to the training directory for the object
            test_dir_path = os.path.join(modelnet40_dir, obj, "test")  # Path to the training directory for the object

            # Check if the training directory exists to avoid processing non-directory files
            if os.path.isdir(training_dir_path):
                off_files = [f for f in os.listdir(training_dir_path) if f.endswith('.off')]  # List all .off files in the training directory
                
                pbar = tqdm(off_files)
                for off_file in pbar:
                    off_file_path = os.path.join(training_dir_path, off_file)  # Full path to the .off file
                        
                    verts, edges = self.off_to_graph(off_file_path)
                    pbar.set_description(f'{obj}_{off_file[-8:-4]}: {len(edges)}')
                    data = Data(x=torch.ones(verts.shape[0],1), edge_index=edges, pos=verts)
                    
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
        
                    train_data_list.append(data)

            if os.path.isdir(training_dir_path):
                off_files = [f for f in os.listdir(test_dir_path) if f.endswith('.off')]  # List all .off files in the training directory
                
                pbar = tqdm(off_files)
                for off_file in pbar:
                    off_file_path = os.path.join(test_dir_path, off_file)  # Full path to the .off file
                        
                    verts, edges = self.off_to_graph(off_file_path)
                    pbar.set_description(f'{obj}_{off_file[-8:-4]}: {len(edges)}')
                    data = Data(x=torch.ones(verts.shape[0],1), edge_index=edges, pos=verts)
                    
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
        
                    test_data_list.append(data)

            
        data, slices = self.collate(train_data_list)
        torch.save((data, slices), self.processed_paths[0])
        data, slices = self.collate(test_data_list)
        torch.save((data, slices), self.processed_paths[1])
        pass

    def off_to_graph(self, file_path):
        mesh = trimesh.load(file_path)
        verts = mesh.vertices
        edges = mesh.edges_unique
        return verts, edges


if __name__=='__main__':
    modelnet = ModelNet40Dataset(root='./data/modelnet')
