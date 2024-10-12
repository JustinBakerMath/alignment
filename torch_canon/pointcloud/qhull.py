'''
qhull
=====

'''

from abc import ABCMeta

import numpy as np
from scipy.spatial import ConvexHull

import shapely

from shapely.geometry import MultiPoint

def get_ch_graph(data, rank, n, *args, **kwargs):
    if n==1:
        return np.array([])
    elif n==2:
        return np.array([[0,1]])
    elif n==3:
        return np.array([[0,1],[1,2],[2,0]])
    else:
        if rank==2:
            hull = find_convex_hull_2d(data.numpy())
        else:
            hull =  ConvexHull(data,qhull_options='QJ')
        if hull.simplices[0].shape[0]==3:
            edges = set()
            for simplex in hull.simplices:
                for i in range(3):
                    edge = sorted([simplex[i], simplex[(i + 1) % 3]])
                    edges.add(tuple(edge))
            return np.array(list(edges))
        else:
            return hull.simplices



def find_convex_hull_2d(points):
    # Assuming `points` is an (N, 3) array of 3D coordinates
    # Find the normal vector of the plane
    mean_point = np.mean(points, axis=0)
    _, _, Vt = np.linalg.svd(points - mean_point)
    normal = Vt[-1]
    
    # Choose two orthogonal vectors to the normal vector to form a basis
    basis_x = Vt[0]
    basis_y = np.cross(normal, basis_x)
    
    # Project points onto this 2D plane
    points_2d = np.dot(points - mean_point, np.array([basis_x, basis_y]).T)
    
    # Find the convex hull in the 2D plane
    hull = ConvexHull(points_2d, qhull_options='Qs')
    
    return hull
