import numpy as np
from vtkmodules.all import vtkMath
from typing import Sequence

def get_distance_between_points(p1:Sequence[float],p2:Sequence[float]):
    """return euclidean distance between points"""
    return np.sqrt(vtkMath.Distance2BetweenPoints(p1,p2))

def get_angle_between_vectors(v1,v2,degrees=True):
    angle = vtkMath.AngleBetweenVectors(v1,v2)
    if degrees:
        angle = vtkMath.DegreesFromRadians(angle)
    return angle

def get_vector_from_points(p1,p2):
    v = np.subtract(p1,p2)
    v_norm = vtkMath.Norm(v)
    return v / v_norm