'''
3D Alignment
============

Includes:
    - Alignment Transforms
    - Coordinate Transforms
    - Vector Alignment
    - Vector Space Operations
'''

import math
import numpy as np
from scipy.spatial.transform import Rotation
import torch

# Alignment Transforms
# ---------------------
def align_pc_t(pointcloud):
    ref_frame = pointcloud.mean(0) 
    return pointcloud - ref_frame, ref_frame

def align_pc_s3(data, us_data, pth):
        data = data.numpy()
        us_data = us_data.numpy()
        funcs = {0: z_axis_alignment, 1: zy_planar_alignment, 2: sign_alignment}
        frame = torch.eye(3).to(torch.float32)
        for idx,val in enumerate(pth):
            data, rot = funcs[idx](data, us_data[val])
            us_data, rot = funcs[idx](us_data, us_data[val])
            frame = rot @ frame
        return data, frame


#def align_pc_s3(data, ref_frame):
    #funcs = {
            #0: z_axis_alignment, 
            #1: yz_planar_alignment,
            #2: sign_alignment}
    #ref_frame = torch.eye(3)
    #for idx,val in enumerate(pth):
        #data, rot = funcs[idx](data, shell_data[val])
        #shell_data = shell_data*rot
        #ref_frame = ref_frame*rot
    #return data, ref_frame

# Coordinate Transforms
# -----------------------

def cartesian2spherical_xtheta(x, y, z):
  " Return spherical coords with theta from the x-axis"
  cart = torch.tensor([x, y, z], dtype=torch.float32)
  r = torch.norm(cart, p=2, dim=-1)
  theta = torch.acos(cart[..., 0] / r) if x!=0 else torch.pi/2
  phi = torch.atan2(cart[...,1], cart[..., 2])
  return r, theta, phi

def cartesian2spherical_ytheta(x, y, z):
  " Return spherical coords from y-axis"
  cart = torch.tensor([x, y, z], dtype=torch.float32)
  r = torch.norm(cart, p=2, dim=-1)
  theta = torch.acos(cart[..., 1] / r) if y!=0 else torch.pi/2
  phi = torch.atan2(cart[...,0], cart[..., 2])
  return r, theta, phi

def cartesian2spherical_ztheta(x, y, z):
  " Return spherical coords from z-axis"
  cart = torch.tensor([x, y, z], dtype=torch.float32)
  r = torch.norm(cart, p=2, dim=-1)
  theta = torch.acos(cart[..., 2] / r) if z!=0 else torch.pi/2
  phi = torch.atan2(cart[...,0], cart[..., 1])
  return r, theta, phi


# Vector Alignment
# -----------------
def xy_planar_alignment(positions, align_vec):
  " Align vector into xy-plane via rotation about x-axis"
  r,theta,phi = cartesian2spherical_xtheta(*align_vec)
  Q = Rotation.from_euler('x',[phi-torch.pi/2],degrees=False).as_matrix().squeeze()
  Q = torch.from_numpy(Q).to(torch.float32)
  for i, pos in enumerate(positions):
    positions[i] = Q@pos
  return positions, Q

def xz_planar_alignment(positions, align_vec):
  " Align vector into xz-plane via rotation about x-axis"
  r,theta,phi = cartesian2spherical_xtheta(*align_vec)
  Q = Rotation.from_euler('x',[phi],degrees=False).as_matrix().squeeze()
  Q = torch.from_numpy(Q).to(torch.float32)
  for i, pos in enumerate(positions):
    positions[i] = Q@pos
  return positions, Q

def zy_planar_alignment(positions, align_vec):
  " Align vector into zy-plane via rotation about z-axis"
  r,theta,phi = cartesian2spherical_ztheta(*align_vec)
  Q = Rotation.from_euler('z',[phi],degrees=False).as_matrix().squeeze()
  Q = torch.from_numpy(Q).to(torch.float32)
  for i, pos in enumerate(positions):
    positions[i] = Q@pos
  return positions, Q

def z_axis_alignment(positions, align_vec):
  " Align vector with z-axis"
  r,theta,phi = cartesian2spherical_ztheta(*align_vec)
  Qz = Rotation.from_euler('z',[phi],degrees=False).as_matrix().squeeze()
  sign = -1 if align_vec[2]<0 else 1
  Qy = Rotation.from_euler('x',[theta],degrees=False).as_matrix().squeeze()
  Qz = torch.from_numpy(Qz).to(torch.float32)
  Qy = torch.from_numpy(Qy).to(torch.float32)
  for i, pos in enumerate(positions):
    positions[i] = (Qz@pos)
    positions[i] = Qy@(pos)
  return positions, Qy@Qz

def sign_alignment(positions, align_vec):
  " Align vector to positive x-direction"
  val = -1 if align_vec[0]<0 else 1
  positions[:,0] = val * positions[:,0]
  R =  torch.eye(3).to(torch.float32)
  R[0,0] = val
  return positions, R


# Vector Space Operations
# -----------------------
def planar_normal(v0, v1):
    return torch.cross(v0, v1)/torch.linalg.norm(torch.cross(v0, v1))

def project_onto_plane(vectors, plane_normal):
    return vectors - torch.matmul(vectors, plane_normal).unsqueeze(-1)*plane_normal

def angle_between_vectors(v0, v1):
    value = torch.dot(v0, v1)/(torch.linalg.norm(v0)*torch.linalg.norm(v1)+1e-10)
    value = torch.clip(value, -1.0, 1.0)
    return torch.arccos(value)
