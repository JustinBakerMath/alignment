import torch
import numpy as np

from spherical_geometry.great_circle_arc import angle as spherical_angle

def spherical_angles_between_vectors(vec0, vec1, vec2):
    angle = spherical_angle(vec0, vec1, vec2)
    if angle > np.pi:
        return 2*np.pi - angle
    if np.isnan(angle):
        return torch.tensor(0.0)
    return angle

def check_colinear(vec1, vec2, tol):
    """
    Check if two vectors are colinear
    """
    if torch.allclose(torch.linalg.cross(vec1, vec2), torch.zeros_like(vec1), atol=tol, rtol=tol):
        return True
    else:
        return False
