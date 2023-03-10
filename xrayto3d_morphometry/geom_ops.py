import numpy as np
from vtkmodules.all import vtkMath,vtkLine
from typing import Sequence
from .tuple_ops import *

"""thin wrapper around vtkMath"""

def get_distance_between_points(p1:Sequence[float],p2:Sequence[float]):
    """return euclidean distance between points"""
    return np.sqrt(vtkMath.Distance2BetweenPoints(p1,p2))

def get_distance_to_line_segment(p0:Sequence[float],line_p0:Sequence[float],line_p1:Sequence[float]):
    return vtkLine.DistanceToLine(p0,line_p0,line_p1) 

def get_angle_between_vectors(v1,v2,degrees=True):
    angle = vtkMath.AngleBetweenVectors(v1,v2)
    if degrees:
        angle = vtkMath.DegreesFromRadians(angle)
    return angle

def get_vector_from_points(p1,p2):
    """return a vector pointing from p1 to p2 (not p2 to p1, order(direction) is important in vectors)"""
    v = np.subtract(p2,p1)
    v_norm = vtkMath.Norm(v)
    return v / v_norm

def lerp(p0: Sequence[float],p1: Sequence[float],alpha: float):
    """linear interpolation"""
    return tuple( a*alpha + b*(1.0 - alpha) for a,b in zip(p0,p1)) 

def get_points_along_directions(point:Sequence[float],direction:Sequence[float],num_points:int=6):
    candidate_points = []
    for i in np.linspace(-num_points//2,num_points//2,num_points):
        candidate_points.append(add_tuple(point,multiply_tuple_scalar(direction,i)))
    return candidate_points
