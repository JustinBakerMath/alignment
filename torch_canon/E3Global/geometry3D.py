import torch
import numpy as np


def check_colinear(vec1, vec2, tol):
    """
    Check if two vectors are colinear
    """
    if torch.allclose(torch.cross(vec1, vec2), torch.zeros_like(vec1), atol=tol, rtol=tol):
        return True
    else:
        return False
