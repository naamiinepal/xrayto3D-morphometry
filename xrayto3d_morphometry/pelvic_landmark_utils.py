"""hip landmark utils"""
from xrayto3d_morphometry import (
    get_mesh_from_segmentation,
    move_to_origin,
    align_along_principal_axes,
)


def get_pelvic_landmarks(nifti_filename):
    """return pelvic landmarks as key:value pairs"""
    mesh = get_mesh_from_segmentation(nifti_filename)

    move_to_origin(mesh)

    aligned_mesh, T = align_along_principal_axes(mesh)
