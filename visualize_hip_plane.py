from xrayto3d_morphometry import *

def visualize_mesh(nifti_filename,offscreen=False,screenshot_out_dir='.'):
    mesh_obj = get_mesh_from_segmentation(nifti_filename)
    mesh_obj.rotate_x(180,around=mesh_obj.center_of_mass()) # for camera orientation
    move_to_origin(mesh_obj)

    # rotate around principal axis
    aligned_mesh_obj, T = align_along_principal_axes(mesh_obj)
    # Orientation AFTER principal axis transformation has to be:
    #     with respect to patient(which affects left-to-right orientation) assume segmentation volume is in PIR
    #      ______________________________________________________
    #     |    Axes    |      X      |      Y      |      Z      |
    #     |  Positive  |    Left     |    Inferior |   Anterior  |
    #     |  Negative  |    Right    |    Superior |  Posterior  |
    #     |______________________________________________________|
    transverse_axis_normal = (0,1,0) # assume PIR orientation
    sagittal_axis_normal = (1,0,0)

    mwp = get_maximal_pelvic_width(aligned_mesh_obj)
    _,_,_,mwp_midpoint = mwp
    TRANSVERSE_PLANE_ALPHA = 0.6
    tph, distal_midpoint = get_transverse_plane_height(aligned_mesh_obj,mwp_midpoint,alpha=TRANSVERSE_PLANE_ALPHA)    
    asis_points = get_asis_estimate(aligned_mesh_obj,transverse_plane_pos=(0,tph,0))
    psis_points = get_psis_estimate(aligned_mesh_obj,transverse_plane_pos=(0,tph,0))
    asis_p1,asis_p2,pt_p1,pt_p2,ps,_ = asis_points
    asis_midpoint = get_midpoint(asis_p1,asis_p2)
    app_height = lerp(ps.GetPosition(),asis_midpoint.GetPosition(),0.5)[1]
    ps_height =ps.GetPosition()[1]

    # Sanity Check: for some cases, aligning along principal axis results in flipped meshes, bring them back to correct orientation by checking ASIS and MWP orientation
    if asis_midpoint.GetPosition()[1] < mwp_midpoint.GetPosition()[1]:
        if 's0477' in nifti_filename:
            pass
        else:
            aligned_mesh_obj.rotate_x(angle=180,around=(0,0,0))
            # redo ASIS
            mwp = get_maximal_pelvic_width(aligned_mesh_obj)
            _,_,_,mwp_midpoint = mwp
            TRANSVERSE_PLANE_ALPHA = 0.6
            tph, distal_midpoint = get_transverse_plane_height(aligned_mesh_obj,mwp_midpoint,alpha=TRANSVERSE_PLANE_ALPHA)    
            asis_points = get_asis_estimate(aligned_mesh_obj,transverse_plane_pos=(0,tph,0))
            psis_points = get_psis_estimate(aligned_mesh_obj,transverse_plane_pos=(0,tph,0))
            asis_p1,asis_p2,pt_p1,pt_p2,ps,_ = asis_points
            asis_midpoint = get_midpoint(asis_p1,asis_p2)
            app_height = lerp(ps.GetPosition(),asis_midpoint.GetPosition(),0.5)[1]
            ps_height =ps.GetPosition()[1]

    # get posterior points for ischial spine detection
    ischial_mesh_left, ischial_mesh_right = get_ischial_mesh_cut(aligned_mesh_obj,ps_height,app_height)
    ischial_spines_left = get_candidate_ischial_spines(ischial_mesh_left)
    ischial_spines_right = get_candidate_ischial_spines(ischial_mesh_right)
    is_1, is_2,is_dist = brute_force_search_get_closest_points_between_point_clouds(
        [p.GetPosition() for p in ischial_spines_left],
        [p.GetPosition() for p in ischial_spines_right]
    )

    psis_p1,psis_p2,msp_p1,msp_p2,top_left_mesh,top_right_mesh = psis_points
    # get sacral promontory by cutting mesh along msp_p1 and msp_p2
    sp_found=True
    try:
        sacral_region, sp = get_sacral_promontory(aligned_mesh_obj, is_1, is_2, msp_p1, msp_p2,asis_midpoint)
        
    except LandmarkNotFoundError as e:
        sp_found = False

    # sanity check: SP should be superior to ASIS
    if sp_found:
        if asis_midpoint.GetPosition()[1] < sp[1]:
            print(nifti_filename)

    cam = get_oriented_camera(aligned_mesh_obj,2,400)
    vedo.show(
        aligned_mesh_obj.c('white',0.6),
        sacral_region.c('green') if sp_found else vedo.Point(alpha=0.0),
        vedo.Point(sp).c('blue') if sp_found else vedo.Point(alpha=0.0),
        # ischial_mesh_left.c('magenta'),
        # ischial_mesh_right.c('blue'),
        # *[ p.c('green') for p in ischial_spines_left],
        # *[ p.c('green') for p in ischial_spines_right],
        vedo.Point(is_1,c='blue',r=18),
        vedo.Point(is_2,c='blue',r=18),
        # vedo.Plane((0,tph,0),transverse_axis_normal,s=(200,200),alpha=0.5),
        vedo.Plane((0,0,0),sagittal_axis_normal,s=(50,50),alpha=0.5),
        # vedo.Plane((0,app_height,0),transverse_axis_normal,s=(200,200),alpha=0.5,c='magenta'),
        # vedo.Plane((0,ps_height,0),transverse_axis_normal,s=(200,200),alpha=0.5,c='magenta'),
        # *[ p.c('blue') for p in mwp],
        *[ p.c('blue') for p in asis_points[:-2]],
        # *[ p.c('blue') for p in psis_points[:-2]],
        # *[p.c(color) for p,color in zip(psis_points[-2:],('blue','red'))],

        resetcam=False,camera = cam,axes=1,offscreen=offscreen)
    out_filename = Path(nifti_filename).with_suffix('.png')
    vedo.screenshot(str(Path(screenshot_out_dir)/out_filename.name))
    if offscreen:
        vedo.close()

def get_sacral_promontory(aligned_mesh_obj, is_1, is_2, msp_p1, msp_p2,asis_midpoint):
    # initial cut plane
    left_sagittal = (msp_p1.GetPosition()[0],0,0)
    right_sagittal = (msp_p2.GetPosition()[0],0,0)
    top_transverse = lerp(is_1,is_2,0.5)
    sacral_region = aligned_mesh_obj.clone(transformed=True).cut_with_plane(left_sagittal,(1,0,0)).cut_with_plane(right_sagittal,(1,0,0),invert=True).cut_with_plane(top_transverse,(0,-1,0))

    sp_idx = float('nan')
    cutting_factor = 0.9

    while math.isnan(sp_idx) and len(sacral_region.points()) != 0:

        sp,sp_idx = get_farthest_point_along_axis(sacral_region.points(),axis=2,negative=False)
        boundary_points = sacral_region.boundaries(return_point_ids=True) 
        # is sp_idx in the boundary or lower than ASIS
        if sp_idx in boundary_points or asis_midpoint.GetPosition()[1] < sp[1]:

            # print(f'{sp} is in boundary')
            sp_idx = float('nan') # is not valid, reset.
            
            # reduce the region
            left_sagittal = (msp_p1.GetPosition()[0]*cutting_factor,0,0)
            right_sagittal = (msp_p2.GetPosition()[0]*cutting_factor,0,0)
            top_transverse = lerp(is_1,is_2,0.5)
            sacral_region = aligned_mesh_obj.clone(transformed=True).cut_with_plane(left_sagittal,(1,0,0)).cut_with_plane(right_sagittal,(1,0,0),invert=True).cut_with_plane(top_transverse,(0,-1,0))
            cutting_factor = cutting_factor -  0.1 # next time reduce more
    if math.isnan(sp_idx):
        raise LandmarkNotFoundError('SP not found')
    return sacral_region,sacral_region.points()[sp_idx]


def get_candidate_ischial_spines(ischial_mesh_left):
    ischial_mesh_left.compute_connectivity()
    ischial_spines = []
    for i in range(-5,45,1): # rotate mesh along x-axis by (-5,45) degree in stepsize of 1 degree
        ischial_mesh_rotated = ischial_mesh_left.clone(transformed=True).rotate_x(i)
        candidate_is, candidate_is_idx = get_farthest_point_along_axis(ischial_mesh_rotated.points(),axis=2,negative=True)
        ischial_spines.append(vedo.Point(ischial_mesh_left.points()[candidate_is_idx]))
    return ischial_spines

def get_landmarks(gt_filename, pca_aligned_mesh):
    mwp_gt = get_maximal_pelvic_width(pca_aligned_mesh)
    _,_,_,anterior_midpoint_gt = mwp_gt
    TRANSVERSE_PLANE_ALPHA = 0.6
    transverse_plane_height_gt,distal_midpoint_gt = get_transverse_plane_height(pca_aligned_mesh, anterior_midpoint_gt,alpha=TRANSVERSE_PLANE_ALPHA)
    gt_points = get_asis_estimate(pca_aligned_mesh,transverse_plane_pos=(0,transverse_plane_height_gt,0))

    # Orientation checks
    # The widest points should be farther along I-S axis than ASIS
    asis_midpoint_gt = get_midpoint(gt_points[0],gt_points[1])
    if anterior_midpoint_gt.GetPosition()[1] < asis_midpoint_gt.GetPosition()[1]:
        if 's0477' in gt_filename:
            pass
        else:
            pca_aligned_mesh.rotate_x(angle=180,around=(0,0,0))

            # redo ASIS estimation 
            mwp_gt = get_maximal_pelvic_width(pca_aligned_mesh)
            _,_,_,anterior_midpoint_gt = mwp_gt

            transverse_plane_height_gt,distal_midpoint_gt = get_transverse_plane_height(pca_aligned_mesh, anterior_midpoint_gt,alpha=TRANSVERSE_PLANE_ALPHA)

            gt_points = get_asis_estimate(pca_aligned_mesh,transverse_plane_pos=(0,transverse_plane_height_gt,0))

    # Sanity check: Pubic Tubercles(PT) should be closer to distal plane than ASIS
    asis_midpoint_gt = get_midpoint(gt_points[0],gt_points[1])
    gt_points = fine_tune_pt_after_sanity_check(gt_filename, pca_aligned_mesh, distal_midpoint_gt, gt_points, asis_midpoint_gt)
    return mwp_gt,transverse_plane_height_gt,distal_midpoint_gt,gt_points

def fine_tune_pt_after_sanity_check(filename, mesh_obj, distal_midpoint, asis_points, asis_midpoint,reduce_by=5):
    pt_1, pt_2 = asis_points[2],asis_points[3]
    pt_1_test_passed = pt_sanity_test(filename, distal_midpoint, asis_midpoint, pt_1)
    pt_2_test_passed = pt_sanity_test(filename,distal_midpoint,asis_midpoint,pt_2)

    if not pt_1_test_passed or not pt_2_test_passed:
        # lower the transverse plane height
        if not pt_1_test_passed: 
            new_lower_transverse_plane_height = pt_1.GetPosition()[1] - reduce_by
        elif not pt_2_test_passed:
            new_lower_transverse_plane_height = pt_2.GetPosition()[1] - reduce_by
        asis_points = get_asis_estimate(mesh_obj,(0,new_lower_transverse_plane_height,0))
    return asis_points

def pt_sanity_test(filename, distal_midpoint, asis_midpoint, pt):
    """returns True if the test passed else False"""
    test_passed = True
    dist_pt_distal = abs(pt.GetPosition()[1] - distal_midpoint.GetPosition()[1])
    dist_pt_asis = abs(pt.GetPosition()[1] - asis_midpoint.GetPosition()[1])
    if dist_pt_distal > dist_pt_asis:
        test_passed = False
        print(test_passed,Path(filename).stem,'PT distance to distal ',dist_pt_distal,'distance to asis ',dist_pt_asis)

    return test_passed 

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('nifti_file')

    args = parser.parse_args()

    visualize_mesh(args.nifti_file)

def process_runs():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('run_id')
    args = parser.parse_args()

    run_id = args.run_id
    # remove old visualizations
    if Path(f'2d-3d-benchmark/{run_id}/visualization_v4').exists():
        shutil.rmtree(Path(f'2d-3d-benchmark/{run_id}/visualization_v4'))
    Path(f'2d-3d-benchmark/{run_id}/visualization_v4').mkdir()

    for f in get_files_from_run_id(run_id=run_id,suffix_regex='*.nii.gz'):
        visualize_mesh(nifti_filename=str(f),offscreen=True,screenshot_out_dir=f'2d-3d-benchmark/{run_id}/visualization_v4')

if __name__ == '__main__':
    # main()
    process_runs()