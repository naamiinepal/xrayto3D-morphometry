import vedo
import numpy as np
from typing import Tuple,List
import SimpleITK as sitk

def get_principal_axis(mesh_obj:vedo.Mesh) -> Tuple[np.ndarray,vedo.Ellipsoid]:
    mesh_axes:vedo.Ellipsoid = vedo.pca_ellipsoid(mesh_obj.points())

    ax1 = vedo.versor(mesh_axes.axis1)
    ax2 = vedo.versor(mesh_axes.axis2)
    ax3 = vedo.versor(mesh_axes.axis3)
    T = np.array([ax1,ax2,ax3])
    return T,mesh_axes

def get_furthest_point_along_axis(point_list: np.ndarray,axis:int,negative:bool=False):
    """find the coordinates and index of the point at furthest distance along an axis"""
    if negative:
        asis_index = np.argmin(point_list[:,abs(axis)])
    else:
        asis_index = np.argmax(point_list[:,abs(axis)])
    return point_list[asis_index],asis_index

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
    return mesh_obj.shift(*-mesh_obj.center_of_mass())