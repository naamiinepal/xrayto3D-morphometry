from xrayto3d_morphometry import *
import numpy as np

def process_femur(nifti_filename,offscreen=False,screenshot_out_dir='.'):
    mesh_obj = get_mesh_from_segmentation(nifti_filename,largest_component=False)

    label_dict = {'head':1,'neck':2,'intra_troc':3,'sub_troc':4}
    subtroc_mesh = extract_volume_surface(get_segmentation_volume(nifti_filename,label_dict['sub_troc']))
    # diaphysis axis
    diaphysis_com = vedo.Point(subtroc_mesh.center_of_mass())
    initial_diaphysis_cpn,csa = grid_search_candidate_cut_plane(mesh_obj,diaphysis_com.GetPosition(),(0,1,0),verbose=True)

    # ##----use the initial diaphysis axis to traverse through
    # additional_diaphysis_com = get_points_along_directions(diaphysis_com.GetPosition(),initial_diaphysis_cpn)
    # additional_diaphysis_cpn = [grid_search_candidate_cut_plane(mesh_obj,com,initial_diaphysis_cpn,range_min=-0.1,range_max=0.1) for com in additional_diaphysis_com]
    
    # neck axis
    neck_mesh = extract_volume_surface(get_segmentation_volume(nifti_filename,label_dict['neck']))
    neck_com = vedo.Point(neck_mesh.center_of_mass())
    initial_neck_cpn,csa = grid_search_candidate_cut_plane(mesh_obj,neck_com.GetPosition(),(1,0,0),verbose=True)
    ##----use the initial neck axis to traverse through
    # additional_neck_com = get_points_along_directions(neck_com.GetPosition(),initial_neck_cpn)
    # additional_neck_cpn = [grid_search_candidate_cut_plane(mesh_obj,com,initial_neck_cpn,range_min=-0.1,range_max=0.1,verbose=True) for com in additional_neck_com]


    # head sphere
    head_mesh = extract_volume_surface(get_segmentation_volume(nifti_filename,label_dict['head']))
    head_com = vedo.Point(head_mesh.center_of_mass())
    head_sph = vedo.fit_sphere(head_mesh)

    vedo.show(
            mesh_obj.opacity(0.05),
            head_mesh.opacity(0.3).c('blue'),
            neck_mesh.opacity(0.3).c('green'),
            subtroc_mesh.opacity(0.3).c('gray'),
            diaphysis_com,
            mesh_obj.clone().cut_with_plane(diaphysis_com.GetPosition(),initial_diaphysis_cpn).boundaries(),
            get_arrow_actor(diaphysis_com.GetPosition(),initial_diaphysis_cpn).opacity(0.3),
            #   vedo.Points(additional_diaphysis_com,r=10),
            #   *[get_arrow_actor(c,n).opacity(0.1) for c,(n,_) in zip(additional_diaphysis_com,additional_diaphysis_cpn)],
            #   *[mesh_obj.clone().cut_with_plane(c,n).boundaries() for c,(n,_) in zip(additional_diaphysis_com,additional_diaphysis_cpn)],
            neck_com,
              mesh_obj.clone().cut_with_plane(neck_com.GetPosition(),initial_neck_cpn).boundaries(),
              get_arrow_actor(neck_com.GetPosition(),initial_neck_cpn).opacity(0.3),
            #   vedo.Points(additional_neck_com,r=10),
            #   *[get_arrow_actor(c,n).opacity(0.1) for c,(n,_) in zip(additional_neck_com,additional_neck_cpn)],
            #   *[mesh_obj.clone().cut_with_plane(c,n).boundaries() for c,(n,_) in zip(additional_neck_com,additional_neck_cpn)],
              head_com,
              head_sph.opacity(0.3),
              resetcam=False,camera=get_oriented_camera(mesh_obj,2,-400),axes=1,
              offscreen=offscreen,
              zoom=1.5)
    out_filename = Path(nifti_filename).with_suffix('.png')
    vedo.screenshot(str(Path(screenshot_out_dir)/out_filename.name))
    if offscreen:
        vedo.close()


if __name__ == '__main__':
    process_femur(nifti_filename='test_data/s0000_femur_left_msk_detailed_4class.nii.gz',offscreen=False,screenshot_out_dir='test_data')

    # run_id = 'ceio7qj7'
    # # remove old visualizations
    # if Path(f'2d-3d-benchmark/{run_id}/visualization_v2').exists():
    #     shutil.rmtree(Path(f'2d-3d-benchmark/{run_id}/visualization_v2'))
    # Path(f'2d-3d-benchmark/{run_id}/visualization_v2').mkdir()

    # for f in get_files(run_id=run_id,suffix_regex='*.nii.gz'):
    #     process_femur(nifti_filename=str(f),offscreen=True,screenshot_out_dir=f'2d-3d-benchmark/{run_id}/visualization_v2')