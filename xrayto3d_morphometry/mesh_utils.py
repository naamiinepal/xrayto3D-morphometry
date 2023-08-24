from typing import Tuple

import numpy as np
import SimpleITK as sitk
import vedo

from .sitk_utils import make_isotropic
from .geom_ops import lerp


def get_principal_axis(mesh_obj: vedo.Mesh) -> Tuple[np.ndarray, vedo.Ellipsoid]:
    mesh_axes: vedo.Ellipsoid = vedo.pca_ellipsoid(mesh_obj.points())

    ax1 = vedo.versor(mesh_axes.axis1)
    ax2 = vedo.versor(mesh_axes.axis2)
    ax3 = vedo.versor(mesh_axes.axis3)
    T = np.array([ax1, ax2, ax3])
    return T, mesh_axes


def align_along_principal_axes(mesh_obj) -> Tuple[vedo.Mesh, np.ndarray]:
    T, mesh_axis = get_principal_axis(mesh_obj)
    aligned_mesh_obj = mesh_obj.clone(transformed=True).apply_transform(T)

    return aligned_mesh_obj, T


def get_mesh_from_segmentation(filename: str, largest_component=False, flying_edges=True, decimate=False, decimation_ratio=1.0, isosurface_value=1.0, smooth=20, reorient=False, orientation='PIR') -> vedo.Mesh:
    np_volume = get_volume(filename, largest_component, reorient=reorient, orientation=orientation)

    # isosurface_values = get_segmentation_labels(sitk_volume)
    mesh_obj: vedo.Mesh = np_volume.isosurface(value=isosurface_value-0.1, flying_edges=flying_edges)
    mesh_obj = mesh_obj.fill_holes()
    mesh_obj.smooth(niter=smooth)
    if decimate:
        mesh_obj = mesh_obj.decimate(fraction=decimation_ratio)
    return mesh_obj.cap()


def get_volume(filename, largest_component=False, isotropic=True, reorient=False, orientation='PIR') -> vedo.Volume:
    sitk_volume = sitk.ReadImage(filename)
    if reorient:
        sitk_volume = sitk.DICOMOrient(sitk_volume, orientation)

    if largest_component:
        # get largest connected component
        sitk_volume = sitk.RelabelComponent(sitk.ConnectedComponent(
            sitk.Cast(sitk_volume, sitk.sitkUInt8),
        ), sortByObjectSize=True) == 1
    if isotropic:
        sitk_volume = make_isotropic(sitk_volume, 1.0)
    np_volume = vedo.Volume(sitk.GetArrayFromImage(sitk_volume))
    return np_volume


def move_to_origin(mesh_obj: vedo.Mesh):
    """changes the original mesh so that its center of mass lies at (0,0,0)"""
    return mesh_obj.shift(*-mesh_obj.center_of_mass())


def get_pointcloud_from_mesh(mesh_obj: vedo.Mesh, label, label_name='scalars'):
    """get mesh vertices with specific labels"""
    point_cloud: np.ndarray = mesh_obj.clone(transformed=True).points()
    point_labels: np.ndarray = mesh_obj.pointdata[label_name]
    return vedo.Points(point_cloud[point_labels == label])


def get_symmetry_plane(mesh_obj, sampled_points=100):
    '''
    get symmetry plane by mirroring the mesh obj and then registering
    the vertices between the original and mirrored mesh obj.
    Take average direction between 100 random registered points by
    fitting a plane through these sampled points
    '''
    mirrored_vert_mesh = mesh_obj.clone(deep=True, transformed=True).mirror("x")
    mirrored_vert_points = vedo.Points(mirrored_vert_mesh.points())
    vert_mesh_points = vedo.Points(
        mesh_obj.clone(deep=True, transformed=True).points()
    )
    aligned_pts1 = mirrored_vert_points.clone().align_to(vert_mesh_points, invert=False)

    # draw arrows to see where points end up
    rand_idx = np.random.randint(0, len(mesh_obj.points()), sampled_points)
    sampled_vmp = mesh_obj.points()[rand_idx]
    sampled_apts1 = aligned_pts1.points()[rand_idx]
    avg_points = [lerp(a, b, 0.5) for a, b in zip(sampled_vmp, sampled_apts1)]
    sym_plane = vedo.fit_plane(avg_points, signed=True)
    return sym_plane
