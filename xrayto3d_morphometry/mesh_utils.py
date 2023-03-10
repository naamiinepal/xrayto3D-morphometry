import vedo
import numpy as np
from typing import Tuple,List,Union,Sequence
import SimpleITK as sitk
from .geom_ops import get_distance_to_line_segment

def get_principal_axis(mesh_obj:vedo.Mesh) -> Tuple[np.ndarray,vedo.Ellipsoid]:
    mesh_axes:vedo.Ellipsoid = vedo.pca_ellipsoid(mesh_obj.points())

    ax1 = vedo.versor(mesh_axes.axis1)
    ax2 = vedo.versor(mesh_axes.axis2)
    ax3 = vedo.versor(mesh_axes.axis3)
    T = np.array([ax1,ax2,ax3])
    return T,mesh_axes


def get_mesh_from_segmentation(filename:str,largest_component=True,flying_edges=True,decimate=False,decimation_ratio=1.0)->vedo.Mesh:
    sitk_volume = sitk.ReadImage(filename)
    
    if largest_component:
        # get largest connected component
        sitk_volume = sitk.RelabelComponent(sitk.ConnectedComponent(
            sitk.Cast(sitk_volume,sitk.sitkUInt8),
            ),sortByObjectSize=True) == 1
    
    np_volume = vedo.Volume(sitk.GetArrayFromImage(sitk_volume))
    
    # get mesh from isosurface centered at (0,0,0)
    mesh_obj:vedo.Mesh = np_volume.isosurface(flying_edges=flying_edges)
    mesh_obj = mesh_obj.fill_holes()
    if decimate:
        mesh_obj = mesh_obj.decimate(fraction=decimation_ratio)
    return mesh_obj

def move_to_origin(mesh_obj: vedo.Mesh):
    """changes the original mesh so that its center of mass lies at (0,0,0)"""
    return mesh_obj.shift(*-mesh_obj.center_of_mass())

def get_pointcloud_from_mesh(mesh_obj: vedo.Mesh,label,label_name='scalars'):
    """get mesh vertices with specific labels"""
    point_cloud:np.ndarray = mesh_obj.clone(transformed=True).points()
    point_labels:np.ndarray = mesh_obj.pointdata[label_name]
    return vedo.Points(point_cloud[point_labels==label])

def get_closest_point_from_plane(mesh_obj: vedo.Mesh,plane:Union[vedo.Plane,Sequence[float]]):
    return get_extrema_from_plane(mesh_obj,plane,apply_fn=np.argmin)

def get_farthest_point_from_plane(mesh_obj: vedo.Mesh,plane:Union[vedo.Plane,Sequence[float]]):
    return get_extrema_from_plane(mesh_obj,plane,apply_fn=np.argmax)

def get_extrema_from_plane(mesh_obj: vedo.Mesh,plane:Union[vedo.Plane,Sequence[float]],apply_fn):
    """if plane is a vector, treat it as a normal to the plane"""
    if isinstance(plane,Sequence):                       
        plane = vedo.Plane(normal=plane)
    mesh_obj.distance_to(plane)
    candidate_point_idx = apply_fn(mesh_obj.pointdata['Distance'])
    return mesh_obj.points()[candidate_point_idx],candidate_point_idx

def get_farthest_point_from_line_segment(points:np.ndarray,line_p0:Sequence[float],line_p1:Sequence[float]):
    """return the point farthest from line segment represented by start and end point"""
    distance_to_line = [get_distance_to_line_segment(p,line_p0,line_p1) for p in points]
    candidate_point_idx = np.argmax(distance_to_line)
    return points[np.argmax(distance_to_line)],candidate_point_idx

def get_farthest_point_along_axis(points: np.ndarray,axis:int,negative:bool=False):
    """find the coordinates and index of the point at furthest distance along an axis"""
    if negative:
        asis_index = np.argmin(points[:,abs(axis)])
    else:
        asis_index = np.argmax(points[:,abs(axis)])
    return points[asis_index],asis_index