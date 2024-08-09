'''
Encoding 3D
===========

Includes:
    - Unit Sphere (US)
    - Convex Hull (CH)

'''
import torch

from torch_canon.utilities import custom_round, list_rotate
from torch_canon.E3Global.align3D import cartesian2spherical_xtheta, project_onto_plane, angle_between_vectors

# Unit Sphere (US)
#----------------------------
def enc_us_pc(data, tol=1e-16, **kwargs):
    encoding = {}
    dists_hash = {}

    # Project and reduce locally close points (n^2 complexity)
    distances = data.norm(dim=1, keepdim=True)
    proj_data =  data/distances
    close_proj_point_idxs, uq_indx = reduce_us(proj_data, tol=tol)

    # Encode distances
    for i,point_idx in enumerate(close_proj_point_idxs):
        dists = [custom_round(distances[idx],tol) for idx in point_idx]
        dists = tuple(sorted(dists)) # sort distances
        if dists not in dists_hash:
            dists_hash[dists] = id(dists) # Hash data
        encoding[i] = dists_hash[dists]

    return dists_hash, encoding, proj_data[uq_indx]

def enc_us_catpc(data, cat_data, tol=1e-16, **kwargs):
    encoding = {}
    dists_hash = {}

    # Project and reduce locally close points (n^2 complexity)
    distances = data.norm(dim=1, keepdim=True)
    proj_data =  data/distances
    locally_close_idx_arrs, uq_indx = reduce_us(proj_data, tol=tol)

    # Encode information while pooling locally close points
    for i,idx_arr in enumerate(locally_close_idx_arrs):
        dists = [(custom_round(distances[idx],tol), custom_round(cat_data[idx],tol))  for idx in idx_arr] # Collect local info
        dists = tuple(sorted(dists)) # Sort data
        if dists not in dists_hash:
            dists_hash[dists] = id(dists) # Hash data
        encoding[i] = dists_hash[dists]

    return dists_hash, encoding, proj_data[uq_indx]



# Convex Hull (CH)
#----------------------------
def enc_ch_pc(us_data, adj_list, us_rank, tol=1e-16):
    encoding = {}
    g_hash = {}

    # Encode edge information
    for point in adj_list.keys():
        r_ij = us_data[adj_list[point]]-us_data[point]

        # distance encoding
        if us_rank == 1: # 
            d_ij = torch.zeros_like(torch.linalg.norm(r_ij, axis=1))
        else:
            d_ij = torch.linalg.norm(r_ij, axis=1)

        # angle encoding
        projection = project_onto_plane(r_ij, us_data[point])
        angle = []
        for i in range(len(projection)):
            if us_rank == 3:
                angle += [angle_between_vectors(projection[i], projection[i-1])]
            else:
                angle += [0]

        # lexicographical shift
        lst = [(custom_round(a,tol),custom_round(d,tol)) for a,d in zip(angle, d_ij)]
        lst = tuple(list_rotate(lst))
        if lst not in g_hash:
            g_hash[lst] = id(lst)
        encoding[point] = g_hash[lst]
    return g_hash, encoding



# Reduction Tools
#----------------
def reduce_us(us_data, tol=1e-16):
    similar_indices = []
    uq_indices = []
    sph_data = torch.tensor([cartesian2spherical_xtheta(*v) for v in us_data], dtype=torch.float32)
    I = [i for i in range(sph_data.shape[0])]
    while(I):
        i = I[0]
        similar_indices.append([i])
        uq_indices.append(i)
        J = [j for j in I[1:]]
        while(J):
            j=J[0]
            close_theta = torch.isclose(sph_data[i][1], sph_data[j][1], atol=tol, rtol=tol)
            close_gamma = torch.isclose(sph_data[i][2], sph_data[j][2], atol=tol, rtol=tol)
            if close_theta and close_gamma:
                similar_indices[i].append(j)
                I.remove(j)
            J.remove(j)
        I.remove(i)
    return similar_indices, uq_indices
