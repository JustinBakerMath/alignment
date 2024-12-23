'''
Encoding 3D
===========

Includes:
    - Unit Sphere (US)
    - Convex Hull (CH)

'''
#import ipdb
import math
import numpy as np

import torch

from torch_canon.utilities import custom_round, list_rotate
from .align import cartesian2spherical_xtheta, project_onto_plane, angle_between_vectors 
from .complete import check_colinear, spherical_angles_between_vectors

# Unit Sphere (US)
#----------------------------
def enc_us_pc(data, tol=1e-16, **kwargs):
    encoding = {}
    dists_hash = {}

    # Project and reduce locally close points (n^2 complexity)
    distances = data.norm(dim=1, keepdim=True)
    proj_data =  data/distances
    close_proj_point_idxs, uq_indx = reduce_us(proj_data, data, tol=tol)

    # Encode distances
    for i,point_idx in enumerate(close_proj_point_idxs):
        dists = [custom_round(distances[idx],tol) for idx in point_idx]
        dists = tuple(sorted(dists)) # sort distances
        if dists not in dists_hash:
            dists_hash[dists] = len(dists_hash)#id(dists) # Hash data
        encoding[i] = dists_hash[dists]

    return dists_hash, encoding, proj_data[uq_indx]

def enc_us_catpc(data, cat_data, dist_hash=None, dist_encoding=None, tol=1e-16, angle_tol=None, **kwargs):
    encoding = {} if dist_encoding is None else dist_encoding
    dists_hash = {} if dist_hash is None else dist_hash

    # Project and reduce locally close points (n^2 complexity)
    distances = data.norm(dim=1, keepdim=True)
    proj_data =  data/distances
    rank = torch.linalg.matrix_rank(data, rtol=tol, atol=tol)

    if rank==2:
        test_points = reduce_rank2(proj_data, tol=tol)
    else:
        test_points = proj_data

    locally_close_idx_arrs, uq_indx = reduce_us(test_points, data, tol=tol, angle_tol=angle_tol)

    # Encode information while pooling locally close points
    sorted_local_mask = []
    for i,idx_arr in enumerate(locally_close_idx_arrs):
        proj_data[uq_indx[i]] = proj_data[idx_arr].mean(dim=0)
        dists = [(custom_round(distances[idx].item(),tol), custom_round(cat_data[idx].item(),tol))  for idx in idx_arr] # Collect local info
        sorting_index = sorted(range(len(dists)), key=lambda k: dists[k])
        dists = tuple([ dists[k] for k in sorting_index]) # sort distances
        if len(sorting_index) > 1:
            # BUG: there is some issue in the indexing here
            sorted_local_mask.append([idx_arr[k] for k in sorting_index][::-1])
        if dists not in dists_hash:
            dists_hash[dists] = len(dists_hash)#id(dists) # Hash data
        encoding[i] = dists_hash[dists]

    local_mask = [True for _ in range(data.shape[0])]
    for sublist in sorted_local_mask:
        for idx in sublist[1:]:
            local_mask[idx] = False
    return dists_hash, encoding, proj_data[uq_indx], sorted_local_mask, local_mask

# Convex Hull (CH)
#----------------------------
def enc_ch_pc(us_data, edge_dict, us_rank, g_hash=None, g_encoding=None, tol=1e-16, angle_tol=None):
    encoding = {} if g_encoding is None else g_encoding
    g_hash = {} if g_hash is None else g_hash

    angle_tol = 0.03 if angle_tol is None else angle_tol

    # Encode edge information
    for point in edge_dict.keys():
        #ipdb.set_trace()
        # need to get clock-wise order
        r_ij = us_data[edge_dict[point]]#-us_data[point]
        r_ij = r_ij/(r_ij.norm(dim=1, keepdim=True)+1e-16)
        projection = project_onto_plane(r_ij, us_data[point])
        ref_vec = projection.mean(dim=0)
        ref_vec = ref_vec/(ref_vec.norm()+1e-16)
        if ref_vec.norm() < tol:
            #print('ref_vec is zero')
            ref_vec = us_data[~point].mean(dim=0)
        angles = [angle_between_vectors(projection[i], ref_vec).item() for i in range(len(edge_dict[point]))]
        angles = [custom_round(a,angle_tol).item() for a in angles]
        d_ij = [angle_between_vectors(us_data[nbr], us_data[point]) for nbr in edge_dict[point]]
        d_ij = [custom_round(d.item(),tol) for d in d_ij]
        tuples = [(angles[i], d_ij[i], edge_dict[point][i]) for i in range(len(edge_dict[point]))]
        tuples = sorted(tuples, key=lambda x: (x[0], x[1]), reverse=True)
        rot_order = [point for _, _, point in tuples]

        edge_dict[point] = rot_order
        # angle encoding
        angle = []
        for i, idx in enumerate(edge_dict[point]):
            angle.append(spherical_angles_between_vectors(us_data[edge_dict[point][i-1]], us_data[point], us_data[idx], tol=tol))

        d_ij = [angle_between_vectors(us_data[nbr], us_data[point]) for nbr in edge_dict[point]]
        # lexicographical shift
        angles = [custom_round(a.item(),angle_tol) for a in angle]
        dists = [custom_round(d.item(),tol) for d in d_ij]
        #angles, idx = tuple(list_rotate(angles))
        #dists = dists[idx:] + dists[:idx]
        lst = tuple(zip(angles,dists))
        lst,_ = list_rotate(lst)
        if lst not in g_hash:
            g_hash[lst] = len(g_hash) #id(lst)
        encoding[point] = g_hash[lst]
    return g_hash, encoding


# Reduction Tools
#----------------
def reduce_us(us_data, data, tol=1e-16, angle_tol=None):
    angle_tol = 0.03 if angle_tol is None else angle_tol
    similar_indices = []
    uq_indices = []
    I = [i for i in range(us_data.shape[0])]
    while(I):
        i = I[0]
        similar_indices.append([i])
        uq_indices.append(i)
        J = [j for j in I[1:]]
        while(J):
            j=J[0]
            dotij = torch.dot(us_data[i], us_data[j]).clamp(-1,1)
            is_close  = torch.arccos(dotij) < tol
            is_close2 = torch.norm(us_data[i]-us_data[j]) < tol
            if is_close2:
                colinear = check_colinear(data[i], data[j], angle_tol)
                if colinear or is_close:
                    similar_indices[-1].append(j)
                    I.remove(j)
            J.remove(j)
        I.remove(i)
    return similar_indices, uq_indices


def reduce_rank2(points, tol=1e-16):
    points = points.numpy()
    mean_point = np.mean(points, axis=0)
    _, _, Vt = np.linalg.svd(points - mean_point)
    normal = Vt[-1]
    basis_x = Vt[0]
    basis_y = np.cross(normal, basis_x)
    basis_z = np.zeros(3)
    points_3d = np.dot(points - mean_point, np.array([basis_x, basis_y, basis_z]).T)
    return torch.from_numpy(points_3d)
