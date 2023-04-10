"""utils for hip landmark identification"""
import numpy as np
import vedo

from .distances import get_farthest_point_along_axis
from .geom_ops import lerp, get_midpoint


def get_maximal_pelvic_width(hip_mesh_obj):
    """return Maximal pelvic points, width and midpoint
    Maximal Pelvic points are the extreme points where the pelvic region is the widest
    """
    proximal_hip_left, _ = get_farthest_point_along_axis(
        hip_mesh_obj.points(), axis=0, negative=True
    )
    proximal_hip_left = vedo.Point(proximal_hip_left)

    proximal_hip_right, _ = get_farthest_point_along_axis(
        hip_mesh_obj.points(), axis=0, negative=False
    )
    proximal_hip_right = vedo.Point(proximal_hip_right)

    proximal_hip_width = vedo.Line(proximal_hip_left, proximal_hip_right)
    proximal_hip_width_midpoint = get_midpoint(proximal_hip_left, proximal_hip_right)
    return (
        proximal_hip_left,
        proximal_hip_right,
        proximal_hip_width,
        proximal_hip_width_midpoint,
    )


def get_transverse_plane_height(
    mesh_obj, proximal_midpoint, sagittal_axis_normal=(1, 0, 0), alpha=0.6
):
    """return Transverse plane intercept as the midpoint(not exactly, see `alpha` parameter) between proximal midpoint(point joining
    maximal pelvic width points) and distal midpoint"""
    distal_left, distal_left_id = get_farthest_point_along_axis(
        mesh_obj.clone(transformed=True)
        .cut_with_plane(normal=sagittal_axis_normal)
        .points(),
        axis=1,
        negative=False,
    )
    distal_right, distal_right_id = get_farthest_point_along_axis(
        mesh_obj.clone(transformed=True)
        .cut_with_plane(normal=sagittal_axis_normal, invert=True)
        .points(),
        axis=1,
        negative=False,
    )
    distal_midpoint = lerp(distal_left, distal_right, alpha=0.5)
    _, transverse_plane_height, _ = lerp(
        distal_midpoint, proximal_midpoint.GetPosition(), alpha=alpha
    )
    return transverse_plane_height, vedo.Point(pos=distal_midpoint)


def get_psis_estimate(
    hip_mesh_obj,
    transverse_plane_pos,
    transverse_axis_normal=(0, 1, 0),
    sagittal_axis_normal=(1, 0, 0),
    verbose=False,
):
    """TODO: work in progress"""
    top_left_mesh: vedo.Mesh = (
        hip_mesh_obj.clone(transformed=True)
        .cut_with_plane(normal=sagittal_axis_normal)
        .cut_with_plane(
            normal=transverse_axis_normal, origin=transverse_plane_pos, invert=True
        )
    )
    top_right_mesh: vedo.Mesh = (
        hip_mesh_obj.clone(transformed=True)
        .cut_with_plane(normal=sagittal_axis_normal, invert=True)
        .cut_with_plane(
            normal=transverse_axis_normal, invert=True, origin=transverse_plane_pos
        )
    )
    psis_p1 = vedo.Point(
        get_farthest_point_along_axis(top_left_mesh.points(), axis=2, negative=True)[0]
    )
    psis_p2 = vedo.Point(
        get_farthest_point_along_axis(top_right_mesh.points(), axis=2, negative=True)[0]
    )
    #  Get the most superior point (MSP) of each hip bone
    msp_p1 = vedo.Point(
        get_farthest_point_along_axis(top_right_mesh.points(), axis=1, negative=True)[0]
    )
    msp_p2 = vedo.Point(
        get_farthest_point_along_axis(top_left_mesh.points(), axis=1, negative=True)[0]
    )
    # Sanity check PSIS vector: The vector connecting the PSIS points
    # PSIS_vec = get_vector_from_points(psis_p1.GetPosition(), psis_p2.GetPosition())
    # print(PSIS_vec)
    return [psis_p1, psis_p2, msp_p1, msp_p2, top_left_mesh, top_right_mesh]


def get_ischial_mesh_cut(
    hip_mesh_obj,
    ps_height,
    app_height,
    transverse_axis_normal=(0, 1, 0),
    sagittal_plane_pos=(0, 0, 0),
    sagittal_plane_normal=(1, 0, 0),
):
    left: vedo.Mesh = (
        hip_mesh_obj.clone(transformed=True)
        .cut_with_plane(
            origin=(0, ps_height, 0), normal=transverse_axis_normal, invert=True
        )
        .cut_with_plane(origin=(0, app_height, 0), normal=transverse_axis_normal)
        .cut_with_plane(origin=sagittal_plane_pos, normal=sagittal_plane_normal)
    )
    left_largest = left.extract_largest_region()  # remove parts of sacrum

    right: vedo.Mesh = (
        hip_mesh_obj.clone(transformed=True)
        .cut_with_plane(
            origin=(0, ps_height, 0), normal=transverse_axis_normal, invert=True
        )
        .cut_with_plane(origin=(0, app_height, 0), normal=transverse_axis_normal)
        .cut_with_plane(
            origin=sagittal_plane_pos, normal=sagittal_plane_normal, invert=True
        )
    )
    right_largest = right.extract_largest_region()  # remove parts of sacrum

    return left_largest, right_largest


def get_asis_estimate(
    hip_mesh_obj,
    transverse_plane_pos,
    transverse_axis_normal=(0, 1, 0),
    sagittal_axis_normal=(1, 0, 0),
    verbose=False,
):
    """return ASIS(Anterior Superior Illiac Spine) landmarks"""
    bottom_left: vedo.Mesh = (
        hip_mesh_obj.clone(transformed=True)
        .cut_with_plane(normal=sagittal_axis_normal)
        .cut_with_plane(normal=transverse_axis_normal, origin=transverse_plane_pos)
    )
    top_left: vedo.Mesh = (
        hip_mesh_obj.clone(transformed=True)
        .cut_with_plane(normal=sagittal_axis_normal)
        .cut_with_plane(
            normal=transverse_axis_normal, invert=True, origin=transverse_plane_pos
        )
    )
    bottom_right: vedo.Mesh = (
        hip_mesh_obj.clone(transformed=True)
        .cut_with_plane(normal=sagittal_axis_normal, invert=True)
        .cut_with_plane(
            normal=transverse_axis_normal, invert=False, origin=transverse_plane_pos
        )
    )
    top_right: vedo.Mesh = (
        hip_mesh_obj.clone(transformed=True)
        .cut_with_plane(normal=sagittal_axis_normal, invert=True)
        .cut_with_plane(
            normal=transverse_axis_normal, invert=True, origin=transverse_plane_pos
        )
    )

    pt_p1 = vedo.Point(
        get_farthest_point_along_axis(bottom_left.points(), axis=2, negative=False)[0]
    )
    pt_p2 = vedo.Point(
        get_farthest_point_along_axis(bottom_right.points(), axis=2, negative=False)[0]
    )
    asis_p1_ = vedo.Point(
        get_farthest_point_along_axis(top_left.points(), axis=2, negative=False)[0]
    )
    asis_p2 = vedo.Point(
        get_farthest_point_along_axis(top_right.points(), axis=2, negative=False)[0]
    )
    ps: vedo.Points = get_midpoint(pt_p1, pt_p2)
    asis_plane: vedo.Plane = vedo.fit_plane(
        vedo.Points([asis_p1_.GetPosition(), asis_p2.GetPosition(), ps.GetPosition()])
    )

    x_axis = np.asarray(
        tuple(a - b for a, b in zip(asis_p1_.GetPosition(), asis_p2.GetPosition())),
        dtype=float,
    )
    x_axis = x_axis / np.linalg.norm(x_axis)
    if verbose:
        print(f"x axis {x_axis}")
        print(f"y axis {asis_plane.normal}")
    return asis_p1_, asis_p2, pt_p1, pt_p2, ps, asis_plane.opacity(alpha=0.8)
