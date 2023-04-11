import vedo
import numpy as np
from xrayto3d_morphometry import (
    get_vector_from_points,
    get_farthest_point_along_axis,
    lerp,
)


def get_asis_estimate(
    mesh: vedo.Mesh,
    transverse_plane_pos,
    transverse_axis_normal=(0, 1, 0),
    sagittal_plane_pos=(0, 0, 0),
    sagittal_axis_normal=(1, 0, 0),
    verbose=False,
):
    """return ASIS(Anterior Superior Illiac Spine) landmarks"""

    tl, bl, tr, br = get_quadrant_meshes(
        mesh,
        transverse_plane_pos,
        sagittal_plane_pos,
        transverse_axis_normal,
        sagittal_axis_normal,
    )

    pt_p1_coord, pt_p1_idx = get_farthest_point_along_axis(
        bl.points(), axis=2, negative=True
    )
    pt_p2_coord, pt_p2_idx = get_farthest_point_along_axis(
        br.points(), axis=2, negative=True
    )

    asis_p1_coord, asis_p1_idx = get_farthest_point_along_axis(
        tl.points(), axis=2, negative=True
    )
    asis_p2_coord, asis_p2_idx = get_farthest_point_along_axis(
        tr.points(), axis=2, negative=True
    )

    T = get_app_plane_rotation_matrix(
        pt_p1_coord, pt_p2_coord, asis_p1_coord, asis_p2_coord
    )
    if verbose:
        print("Rotation Matrix")
        print(T)
        identity_matrix = np.eye(3)
        print(np.allclose(T, identity_matrix, atol=0.01))

    return (
        mesh.closest_point(pt_p1_coord, return_point_id=True),
        mesh.closest_point(pt_p2_coord, return_point_id=True),
        mesh.closest_point(asis_p1_coord, return_point_id=True),
        mesh.closest_point(asis_p2_coord, return_point_id=True),
        T
    )


def get_app_plane_rotation_matrix(
    pt_p1_coord, pt_p2_coord, asis_p1_coord, asis_p2_coord
):
    """return 3x3 rotation matrix that defines the axes of the APP plane
    defined by the ASIS and Pubic Symphysis"""
    x_direction = get_vector_from_points(pt_p2_coord, pt_p1_coord)
    pt_mid = lerp(pt_p1_coord, pt_p2_coord, 0.5)
    app_points = vedo.Points([asis_p1_coord, asis_p2_coord, pt_mid])
    app_plane = vedo.fit_plane(app_points)
    n = app_plane.normal
    T = np.array([x_direction, np.cross(n, x_direction), n])
    return T


def get_maximal_pelvic_points(hip_mesh):
    """maximal pelvic points along x-axis(left-right axis)"""
    mwp_p1_coord, mwp_p1_idx = get_farthest_point_along_axis(
        hip_mesh.points(), axis=0, negative=True
    )
    mwp2_coord, mwp2_idx = get_farthest_point_along_axis(
        hip_mesh.points(), axis=0, negative=False
    )

    return mwp_p1_idx, mwp2_idx


def get_transverse_plane_height(
    mesh, proximal_midpoint, sagittal_axis_normal=(1, 0, 0), alpha=0.6, verbose=False
):
    """return transverse plane intercept"""
    left_mesh = mesh.clone(transformed=True).cut_with_plane(
        normal=sagittal_axis_normal, origin=(0, 0, 0)
    )
    right_mesh = mesh.clone(transformed=True).cut_with_plane(
        normal=sagittal_axis_normal, origin=(0, 0, 0), invert=True
    )
    distal_left_coord, _ = get_farthest_point_along_axis(
        left_mesh.points(), axis=1, negative=True
    )
    distal_right_coord, _ = get_farthest_point_along_axis(
        right_mesh.points(), axis=1, negative=True
    )
    distal_midpoint = lerp(distal_left_coord, distal_right_coord, alpha=0.5)
    _, transverse_plane_height, _ = lerp(
        distal_midpoint, proximal_midpoint, alpha=alpha
    )
    if verbose:
        return transverse_plane_height, distal_left_coord, distal_right_coord
    return transverse_plane_height


def get_quadrant_meshes(
    hip_mesh,
    transverse_plane_pos=(0, 0, 0),
    sagittal_plane_pos=(0, 0, 0),
    transverse_axis_normal=(0, 1, 0),
    sagittal_axis_normal=(1, 0, 0),
    verbose=False,
):
    """return 4 pieces of meshes cut by transverse_axis and sagittal_axis at given positions"""
    bottom_left: vedo.Mesh = (
        hip_mesh.clone(transformed=True)
        .cut_with_plane(normal=sagittal_axis_normal, origin=sagittal_plane_pos)
        .cut_with_plane(normal=transverse_axis_normal, origin=transverse_plane_pos)
    )

    top_left: vedo.Mesh = (
        hip_mesh.clone(transformed=True)
        .cut_with_plane(
            normal=sagittal_axis_normal, origin=sagittal_plane_pos, invert=False
        )
        .cut_with_plane(
            normal=transverse_axis_normal, origin=transverse_plane_pos, invert=True
        )
    )

    bottom_right: vedo.Mesh = (
        hip_mesh.clone(transformed=True)
        .cut_with_plane(
            normal=sagittal_axis_normal, origin=sagittal_plane_pos, invert=True
        )
        .cut_with_plane(
            normal=transverse_axis_normal, origin=transverse_plane_pos, invert=False
        )
    )

    top_right: vedo.Mesh = (
        hip_mesh.clone(transformed=True)
        .cut_with_plane(
            normal=sagittal_axis_normal, origin=sagittal_plane_pos, invert=True
        )
        .cut_with_plane(
            normal=transverse_axis_normal, origin=transverse_plane_pos, invert=True
        )
    )
    return bottom_left, top_left, bottom_right, top_right
