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

def align_pc_s3(data, shell_data, pth):
        data = data.numpy()
        print(data)
        shell_data = shell_data.numpy()
        funcs = {0: z_axis_alignment, 1: zy_planar_alignment, 2: sign_alignment}
        frame = np.eye(3)
        for idx,val in enumerate(pth):
            data, rot = funcs[idx](data, shell_data[val])
            shell_data, rot = funcs[idx](shell_data, shell_data[val])
            if rot.__class__ == np.ndarray:
                frame = frame @ rot
            else:
                frame = frame*rot
        print(frame)
        print(data.round(2))
        return data, frame

#def align_pc_s3(data, ref_frame):
    #funcs = {
            #0: z_axis_alignment, 
            #1: zy_planar_alignment,
            #2: sign_alignment}
    #ref_frame = torch.eye(3)
    #for idx,val in enumerate(pth):
        #data, rot = funcs[idx](data, shell_data[val])
        #shell_data = shell_data*rot
        #ref_frame = ref_frame*rot
    #return data, ref_frame

# Coordinate Transforms
# -----------------------

def cartesian_to_xspherical(x, y, z):
  " Return spherical coords from x-axis"
  r = np.sqrt(x**2 + y**2 + z**2)
  value = x/(r+1e-10)
  value = np.clip(value, -1.0, 1.0)
  theta = np.arccos(value)
  phi = np.arctan2(z, y)
  return r, theta, phi

def cartesian_to_yspherical(x, y, z):
  " Return spherical coords from y-axis"
  r = np.sqrt(x**2 + y**2 + z**2)
  value = y/(r+1e-10)
  value = np.clip(value, -1.0, 1.0)
  theta = np.arccos(value)
  phi = np.arctan2(z, x)
  return r, theta, phi

def cartesian_to_zspherical(x, y, z):
  " Return spherical coords from z-axis"
  r = np.sqrt(x**2 + y**2 + z**2)
  theta = math.atan2(math.sqrt(x * x + y * y), z)
  phi = math.atan2(y, x)
  return r, theta, phi


# Vector Alignment
# -----------------
def xz_planar_alignment(positions, align_vec):
  " Align vector into xy-plane"
  r,theta,phi = cartesian_to_xspherical(*align_vec)
  Q = Rotation.from_euler('z',[-theta],degrees=False).as_matrix().squeeze()
  for i, pos in enumerate(positions):
    positions[i] = Q@pos
  return positions, Q

def xy_planar_alignment(positions, align_vec):
  " Align vector into xy-plane"
  r,theta,phi = cartesian_to_xspherical(*align_vec)
  Q = Rotation.from_euler('x',[phi],degrees=False).as_matrix().squeeze()
  for i, pos in enumerate(positions):
    positions[i] = Q@pos
  return positions, Q

def z_axis_alignment(positions, align_vec):
  " Align vector with z-axis"
  r,theta,phi = cartesian_to_zspherical(*align_vec)
  Qz = Rotation.from_euler('z',[-phi],degrees=False).as_matrix().squeeze()
  Qy = Rotation.from_euler('y',[-theta],degrees=False).as_matrix().squeeze()
  for i, pos in enumerate(positions):
    positions[i] = Qy@(Qz@pos)
  return positions, Qy@Qz

def zy_planar_alignment(positions, align_vec):
  " Align vector into zy-plane"
  r,theta,phi = cartesian_to_zspherical(*align_vec)
  Q = Rotation.from_euler('z',[-phi],degrees=False).as_matrix().squeeze()
  for i, pos in enumerate(positions):
    positions[i] = Q@pos
  return positions, Q

def sign_alignment(positions, align_vec):
  " Align vector to positive x-direction"
  val = -1 if align_vec[0]<0 else 1
  positions[:,1] = -positions[:,1]
  return positions, val


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
