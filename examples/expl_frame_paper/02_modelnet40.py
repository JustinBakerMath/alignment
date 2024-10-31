#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from tqdm import tqdm

sys.path.append('./examples/datasets/')
from modelnet40 import ModelNet40Dataset
from torch_canon.pointcloud import CanonEn as Canon

train_ds = ModelNet40Dataset(root='./data/modelnet')
test_ds = ModelNet40Dataset(root='./data/modelnet', name='test')

frame = Canon(tol=1e-2)

print(train_ds)
print(test_ds)

for i in tqdm(range(10)):
    print(train_ds[i])
    frame.get_frame(train_ds[i].pos, train_ds[i].x)
