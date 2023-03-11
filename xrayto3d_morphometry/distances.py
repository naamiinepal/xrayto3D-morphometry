from .geom_ops import get_distance_to_line_segment
import vedo
from typing import Union,Sequence
import numpy as np
from .tuple_ops import subtract_tuple,multiply_tuple_scalar,add_tuple

def get_closest_point_from_line(p0:Sequence[float],line_p0: Sequence[float],line_p1: Sequence[float]):
    line_p0_numpy = np.asarray(line_p0,dtype=np.float32)
    line_p1_numpy = np.asarray(line_p1,dtype=np.float32)
    p0_numpy = np.asarray(p0,dtype=np.float32)

    ap = p0_numpy-line_p0_numpy
    ab = line_p1_numpy-line_p0_numpy
    result = line_p0_numpy + np.dot(ap,ab)/np.dot(ab,ab) * ab
    return result.tolist()

def get_closest_point_from_plane(mesh_obj: vedo.Mesh,plane:Union[vedo.Plane,Sequence[float]]):
    return get_extrema_from_plane(mesh_obj,plane,apply_fn=np.argmin)

def get_farthest_point_from_plane(mesh_obj: vedo.Mesh,plane:Union[vedo.Plane,Sequence[float]]):
    return get_extrema_from_plane(mesh_obj,plane,apply_fn=np.argmax)

def get_farthest_point_from_line_segment(points:np.ndarray,line_p0:Sequence[float],line_p1:Sequence[float]):
    """return the point farthest from line segment represented by start and end point"""
    distance_to_line = [get_distance_to_line_segment(p,line_p0,line_p1) for p in points]
    candidate_point_idx = np.argmax(distance_to_line)
    return points[np.argmax(distance_to_line)],candidate_point_idx

def get_line_segment(center:Sequence[float],direction:Sequence[float],distance=400):
    if isinstance(center,vedo.Points):
        center = center.GetPosition()
    if isinstance(direction,vedo.Points):
        direction = direction.GetPosition()

    line_p0 = add_tuple(center,multiply_tuple_scalar(direction,distance))
    line_p1 = subtract_tuple(center,multiply_tuple_scalar(direction,distance))
    return line_p0,line_p1

def get_farthest_point_along_axis(points: np.ndarray,axis:int,negative:bool=False):
    """find the coordinates and index of the point at furthest distance along an axis"""
    if negative:
        candidate_point_idx = np.argmin(points[:,abs(axis)])
    else:
        candidate_point_idx = np.argmax(points[:,abs(axis)])
    return points[candidate_point_idx],candidate_point_idx

def get_extrema_from_plane(mesh_obj: vedo.Mesh,plane:Union[vedo.Plane,Sequence[float]],apply_fn):
    """if plane is a vector, treat it as a normal to the plane"""
    if isinstance(plane,Sequence):                       
        plane = vedo.Plane(normal=plane)
    mesh_obj.distance_to(plane)
    candidate_point_idx = apply_fn(mesh_obj.pointdata['Distance'])
    return mesh_obj.points()[candidate_point_idx],candidate_point_idx