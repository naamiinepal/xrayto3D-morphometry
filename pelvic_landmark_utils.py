import vedo
from xrayto3d_morphometry import get_farthest_point_along_axis


def get_maximal_pelvic_points(hip_mesh):
    mwp_p1_coord, mwp_p1_idx = get_farthest_point_along_axis(
        hip_mesh.points(), axis=0, negative=True
    )
    mwp2_coord, mwp2_idx = get_farthest_point_along_axis(
        hip_mesh.points(), axis=0, negative=False
    )

    return mwp_p1_idx, mwp2_idx


def get_quadrant_meshes(
    hip_mesh,
    transverse_plane_pos=(0, 0, 0),
    sagittal_axis_pos=(0, 0, 0),
    transverse_axis_normal=(0, 1, 0),
    sagittal_axis_normal=(1, 0, 0),
    verbose=False,
):
    """return 4 pieces of meshes cut by transverse_axis and sagittal_axis at given positions"""
    bottom_left: vedo.Mesh = (
        hip_mesh.clone(transformed=True)
        .cut_with_plane(normal=sagittal_axis_normal, origin=sagittal_axis_pos)
        .cut_with_plane(normal=transverse_axis_normal, origin=transverse_plane_pos)
    )

    top_left: vedo.Mesh = (
        hip_mesh.clone(transformed=True)
        .cut_with_plane(
            normal=sagittal_axis_normal, origin=sagittal_axis_pos, invert=False
        )
        .cut_with_plane(
            normal=transverse_axis_normal, origin=transverse_plane_pos, invert=True
        )
    )

    bottom_right: vedo.Mesh = (
        hip_mesh.clone(transformed=True)
        .cut_with_plane(
            normal=sagittal_axis_normal, origin=sagittal_axis_pos, invert=True
        )
        .cut_with_plane(
            normal=transverse_axis_normal, origin=transverse_plane_pos, invert=False
        )
    )

    top_right: vedo.Mesh = (
        hip_mesh.clone(transformed=True)
        .cut_with_plane(
            normal=sagittal_axis_normal, origin=sagittal_axis_pos, invert=True
        )
        .cut_with_plane(
            normal=transverse_axis_normal, origin=transverse_plane_pos, invert=True
        )
    )
    return bottom_left, top_left, bottom_right, top_right
