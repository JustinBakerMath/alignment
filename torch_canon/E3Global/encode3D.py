'''
Encoding 3D
===========

Includes:
    - Unit Sphere (US)
    - Convex Hull (CH)

'''
#import ipdb
import torch

from torch_canon.utilities import custom_round, list_rotate
from torch_canon.E3Global.align3D import cartesian2spherical_xtheta, project_onto_plane, angle_between_vectors 
from torch_canon.E3Global.geometry3D import check_colinear, spherical_angles_between_vectors

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

def enc_us_catpc(data, cat_data, dist_hash=None, dist_encoding=None, tol=1e-16, **kwargs):
    encoding = {} if dist_encoding is None else dist_encoding
    dists_hash = {} if dist_hash is None else dist_hash

    # Project and reduce locally close points (n^2 complexity)
    distances = data.norm(dim=1, keepdim=True)
    proj_data =  data/distances
    locally_close_idx_arrs, uq_indx = reduce_us(proj_data, data, tol=tol)

    # Encode information while pooling locally close points
    for i,idx_arr in enumerate(locally_close_idx_arrs):
        proj_data[uq_indx[i]] = proj_data[idx_arr].mean(dim=0)
        dists = [(custom_round(distances[idx].item(),tol), custom_round(cat_data[idx].item(),tol))  for idx in idx_arr] # Collect local info
        dists = tuple(sorted(dists)) # Sort data
        if dists not in dists_hash:
            dists_hash[dists] = len(dists_hash)#id(dists) # Hash data
        #else:
            #print('Duplicate')
        encoding[i] = dists_hash[dists]

    return dists_hash, encoding, proj_data[uq_indx]

# Convex Hull (CH)
#----------------------------
def enc_ch_pc(us_data, edge_dict, us_rank, g_hash=None, g_encoding=None, tol=1e-16):
    encoding = {} if g_encoding is None else g_encoding
    g_hash = {} if g_hash is None else g_hash

    # Encode edge information
    for point in edge_dict.keys():
        #ipdb.set_trace()
        # need to get clock-wise order
        r_ij = us_data[edge_dict[point]]#-us_data[point]
        r_ij = r_ij/(r_ij.norm(dim=1, keepdim=True)+1e-16)
        projection = project_onto_plane(r_ij, us_data[point])
        ref_vec = projection.mean(dim=0)
        ref_vec = ref_vec/(ref_vec.norm()+1e-16)
        angles = [angle_between_vectors(projection[i], ref_vec).item() for i in range(len(edge_dict[point]))]
        angles = [custom_round(a,tol) for a in angles]
        tuples = [(angles[i],edge_dict[point][i]) for i in range(len(edge_dict[point]))]
        rot_order = [point for _, point in sorted(tuples, key=lambda x: (x[0], x[1]), reverse=True)]

        edge_dict[point] = rot_order
        # angle encoding
        angle = []
        for i, idx in enumerate(edge_dict[point]):
            angle.append(spherical_angles_between_vectors(us_data[edge_dict[point][i-1]], us_data[point], us_data[idx], tol=tol))

        d_ij = [angle_between_vectors(us_data[nbr], us_data[point]) for nbr in edge_dict[point]]
        # lexicographical shift
        angles = [custom_round(a.item(),tol) for a in angle]
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
def reduce_us(us_data, data, tol=1e-16):
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
            if is_close:
                colinear = check_colinear(data[i], data[j], tol)
                if colinear:
                    similar_indices[-1].append(j)
                    I.remove(j)
            J.remove(j)
        I.remove(i)
    return similar_indices, uq_indices
