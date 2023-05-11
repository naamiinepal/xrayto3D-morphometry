"""hip landmark extraction script"""
import argparse
from pathlib import Path
import vedo

from xrayto3d_morphometry import (
    align_along_principal_axes,
    get_asis_estimate,
    get_quadrant_cuts,
    get_ischial_points_estimate,
    get_maximal_pelvic_width,
    get_mesh_from_segmentation,
    get_midpoint,
    get_oriented_camera,
    get_transverse_plane_height,
    move_to_origin,
    get_app_plane_rotation_matrix,
    get_farthest_point_along_axis,
    get_distance_between_points
)

def get_landmarks(mesh_obj, mesh_filename):
    """return landmarks as dict"""

    move_to_origin(mesh_obj)
    # rotate around principal axis
    aligned_mesh_obj, T = align_along_principal_axes(mesh_obj)
    #      ______________________________________________________
    #     |    Axes    |      X      |      Y      |      Z      |
    #     |  Positive  |    Left     |    Inferior |   Anterior  |
    #     |  Negative  |    Right    |    Superior |  Posterior  |
    #     |______________________________________________________|    
    mwp_midpoint = get_maximal_pelvic_width(aligned_mesh_obj)[-1]
    tph = get_transverse_plane_height(aligned_mesh_obj, mwp_midpoint, alpha=0.6)[0]
    bottom_left, top_left, bottom_right, top_right = get_quadrant_cuts(aligned_mesh_obj, transverse_plane_pos=(0, tph, 0))

    asis_p1, asis_p2, pt_p1, pt_p2, ps, asis_plane = get_asis_estimate(bottom_left, top_left, bottom_right, top_right)
    # sanity check: for some cases, aligning along principal axes results
    # in change in mesh orientation, bring them back to correct orientation
    # by checking ASIS and MWP orientation
    asis_midpoint = get_midpoint(asis_p1, asis_p2)
    asis_x, asis_y, asis_z = asis_midpoint.GetPosition()
    mwp_x, mwp_y, mwp_z = mwp_midpoint.GetPosition()
    redo_asis_estimate = False
    if asis_y < mwp_y:
        if "s0477" in mesh_filename:
            # hard-coded failure cases
            pass
        else:
            aligned_mesh_obj.rotate_x(angle=180, around=aligned_mesh_obj.center_of_mass())
        redo_asis_estimate = True
    if redo_asis_estimate:
        mwp_midpoint = get_maximal_pelvic_width(aligned_mesh_obj)[-1]
        tph = get_transverse_plane_height(aligned_mesh_obj, mwp_midpoint, alpha=0.6)[0]
        bottom_left, top_left, bottom_right, top_right = get_quadrant_cuts(aligned_mesh_obj, transverse_plane_pos=(0, tph, 0))
        
        asis_p1, asis_p2, pt_p1, pt_p2, ps, asis_plane = get_asis_estimate(bottom_left, top_left, bottom_right, top_right)

    # second iteration: apply transformation and get asis estimate
    T = get_app_plane_rotation_matrix(pt_p1.GetPosition(), pt_p2.GetPosition(), asis_p1.GetPosition(), asis_p2.GetPosition())
    aligned_mesh_obj.apply_transform(T, concatenate=True)
    bottom_left.apply_transform(T, concatenate=True)
    top_left.apply_transform(T, concatenate=True)
    bottom_right.apply_transform(T, concatenate=True)
    top_right.apply_transform(T, concatenate=True)
    asis_p1, asis_p2, pt_p1, pt_p2, ps, asis_plane = get_asis_estimate(bottom_left, top_left, bottom_right, top_right)

    # get ischial points
    ps_x, ps_y, ps_z = ps.GetPosition()
    asis_midpoint = get_midpoint(asis_p1, asis_p2)
    app_x, app_y, app_z = get_midpoint(ps, asis_midpoint).GetPosition()
    is_1, is_2 = get_ischial_points_estimate(aligned_mesh_obj, ps_y, app_y)

    # get PSIS points
    cut_plane_origin = (0,asis_p1.GetPosition()[1],0)
    superior_boundary = get_farthest_point_along_axis(top_left.points(),axis=1, negative=True)[0]
    print('SUPERIOR Boundary', superior_boundary)
    while True:
        top_left_cut = aligned_mesh_obj.clone(transformed=True).cut_with_plane(normal=(0,1,0), origin=cut_plane_origin, invert=True).cut_with_plane(normal=(1,0,0))
        top_right_cut = aligned_mesh_obj.clone(transformed=True).cut_with_plane(normal=(0,1,0), origin=cut_plane_origin,invert=True).cut_with_plane(normal=(1,0,0),invert=True)
        try:
            psis_p1 = vedo.Point(
                get_farthest_point_along_axis(top_left_cut.points(), axis=2, negative=True)[0]
            )
            psis_p2 = vedo.Point(
                get_farthest_point_along_axis(top_right_cut.points(), axis=2, negative=True)[0]
            )
        except ValueError:
            break

        if ((abs(psis_p1.GetPosition()[0]) < 10) or (abs(psis_p2.GetPosition()[0]) < 10)) or (get_distance_between_points(psis_p1.GetPosition(),psis_p2.GetPosition()) < 20) or (abs(cut_plane_origin[1]-5) >= abs(superior_boundary[1])):
            cut_plane_origin = (0,cut_plane_origin[1]-2,0)
        else:
            break    

    #  return coordinates in original mesh space, not aligned mesh space
    asis_p1_idx = aligned_mesh_obj.closest_point(asis_p1.GetPosition(), return_point_id=True)
    asis_p2_idx = aligned_mesh_obj.closest_point(asis_p2.GetPosition(), return_point_id=True)
    pt_p1_idx = aligned_mesh_obj.closest_point(pt_p1.GetPosition(), return_point_id=True)
    pt_p2_idx = aligned_mesh_obj.closest_point(pt_p2.GetPosition(), return_point_id=True)
    is_p1_idx = aligned_mesh_obj.closest_point(is_1, return_point_id=True)
    is_p2_idx = aligned_mesh_obj.closest_point(is_2, return_point_id=True)
    psis_p1_idx = aligned_mesh_obj.closest_point(psis_p1.GetPosition(), return_point_id=True)
    psis_p2_idx = aligned_mesh_obj.closest_point(psis_p2.GetPosition(), return_point_id=True)

    return {
        "ASIS_L": asis_p1_idx,
        "ASIS_R": asis_p2_idx,
        "PT_L":   pt_p1_idx,
        "PT_R":   pt_p2_idx,
        "IS_L": is_p1_idx,
        "IS_R": is_p2_idx,
        "PSIS_L": psis_p1_idx,
        "PSIS_R": psis_p2_idx,
    }, aligned_mesh_obj


def main(nifti_filename, offscreen=False, screenshot_out_dir="./screenshots"):
    mesh_obj = get_mesh_from_segmentation(nifti_filename)
    mesh_obj.rotate_x(180, around=mesh_obj.center_of_mass())
    
    landmark_indices, aligned_mesh_obj = get_landmarks(mesh_obj, nifti_filename)
    landmarks = [mesh_obj.points()[landmark_indices[key]] for key in landmark_indices]
    print(landmarks)
    # visualize landmarks
    cam = get_oriented_camera(mesh_obj, axis=2, camera_dist=400)
    cam['position'][2] = cam['position'][2]
    vedo.show(
        mesh_obj.c('white', 0.6),
        vedo.Points(landmarks, c='blue', r=24),
        resetcam=False,
        camera=cam,
        axes=1,
        offscreen=offscreen
    )
    out_filename = Path(nifti_filename).with_suffix(".png")
    vedo.screenshot(str(Path(screenshot_out_dir) / out_filename.name))
    if offscreen:
        vedo.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('nifti_file')
    parser.add_argument('--offscreen', default=False, action='store_true')

    args = parser.parse_args()

    main(args.nifti_file, args.offscreen)
