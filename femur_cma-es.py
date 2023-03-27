import vedo
from pathlib import Path
import numpy as np
from xrayto3d_morphometry import (cma_es_search_candidate_cut_plane,
                                  extract_volume_surface, get_line_segment,
                                  get_mesh_from_segmentation,
                                  get_segmentation_volume,
                                  get_farthest_point_from_line_segment,
                                  get_points_along_directions,
                                  get_angle_between_vectors,
                                  get_distance_to_line_segment,
                                  get_vector_from_points,
                                  get_arrow_actor,
                                  get_closest_point_from_line
                                )
                                  

def get_femur_morphometry(nifti_filename,robust=False,visualize=True,screenshot=True,offscreen=True,screenshot_out_dir='.'):
    mesh_obj = get_mesh_from_segmentation(nifti_filename,largest_component=True)
    label_dict = {'head':4,'neck':3,'sub_troc':2}
    subtroc_mesh = extract_volume_surface(
        get_segmentation_volume(nifti_filename,label_dict['sub_troc']))
    
    # diaphysis axis
    diaphysis_center = vedo.Point(subtroc_mesh.center_of_mass()) 
    diaphysis_es = cma_es_search_candidate_cut_plane(mesh_obj,diaphysis_center.GetPosition(),(0,1,0))
    diaphysis_axis,f,evals = diaphysis_es.best.get()
    
  
    # heuristics: Cerveri et.al 2010
    # l_a: line representing diaphysis axis
    # p_c: point on the surface at maximum distance from l_a
    # pi_a: plane passing through p_c and l_a
    # p_m: projection of femur center on l_a
    # l_b: line connecting p_c and p_m
    # pi_c: tangent plane at p_c
    l_a = get_line_segment(diaphysis_center.GetPosition(),diaphysis_axis,400)
    p_c, p_c_idx = get_farthest_point_from_line_segment(mesh_obj.points(),*l_a) # type: ignore
    pi_a = vedo.fit_plane(vedo.Points([*l_a,p_c]))
    p_m = get_closest_point_from_line(mesh_obj.center_of_mass(),*l_a) # type: ignore
    l_b = (p_m, p_c)
    pi_c_normal = get_vector_from_points(p_c,p_m)
    pi_c = vedo.Plane(p_c,pi_c_normal,s=(100,100))

    # get potential femoral head surface points, then fit femoral head
    pc_points = get_points_along_directions(p_c,pi_c_normal,50,positive_only=True)
    candidate_femoral_head_cuts = [mesh_obj.clone().cut_with_plane(p,pi_c_normal).boundaries() for p in pc_points]
    candidate_sphere_points = []
    for cut in candidate_femoral_head_cuts:
        candidate_sphere_points.extend(cut.points().tolist())
    head_sphere:vedo.Sphere = vedo.fit_sphere(candidate_sphere_points) # type: ignore


    # fit femoral neck
    l_n = get_vector_from_points(head_sphere.center,p_m)
    p_n = lerp(head_sphere.center.tolist(),p_m,alpha=0.5)
    neck_es = cma_es_search_candidate_cut_plane(mesh_obj,p_n,l_n,verbose=False)
    neck_normal, f, evals = neck_es.best.get()

    femoral_head_radius = head_sphere.radius # why are we getting radius that is twice as big as it should be ?
    # a normal femoral head is around 16 mm.
    femoral_head_radius = femoral_head_radius / 2 # quick fix: just divide by 2
    if robust:
        #----use the initial diaphysis axis to traverse through
        additional_diaphysis_center = get_points_along_directions(diaphysis_center.GetPosition(),diaphysis_axis)
        additional_diaphysis_axis = [cma_es_search_candidate_cut_plane(mesh_obj,center,diaphysis_axis).best.get()[0] for center in additional_diaphysis_center]    
        additional_diaphysis_directions = [get_vector_from_points(*get_line_segment(p,n,10)) for p,n in zip(additional_diaphysis_center,additional_diaphysis_axis)]
        additional_ct_x_angles = [get_angle_between_vectors(n,(1,0,0)) for n in additional_diaphysis_directions]
        additional_ct_y_angles = [get_angle_between_vectors(n,(0,1,0)) for n in additional_diaphysis_directions]
        additional_ct_z_angles = [get_angle_between_vectors(n,(0,0,1)) for n in additional_diaphysis_directions]

        additional_l_a = [get_line_segment(center,axis,400) for center,axis in zip(additional_diaphysis_center,additional_diaphysis_axis)]
        additional_femoral_head_offset = [get_distance_to_line_segment(head_sphere.center.tolist(),*l) / 2 for l in additional_l_a]
        mean_fho = np.mean(additional_femoral_head_offset)
        std_fho = np.std(additional_femoral_head_offset)

        #----use the initial neck axis to traverse through
        additional_p_n = get_points_along_directions(p_n,neck_normal)
        additional_neck_pi_n = np.asarray([cma_es_search_candidate_cut_plane(mesh_obj,com,neck_normal).best.get()[0] for com in additional_p_n])
        additional_neck_directions = [get_vector_from_points(*get_line_segment(p,n,10)) for p,n in zip(additional_p_n,additional_neck_pi_n)]
        additional_neck_x_angles = [get_angle_between_vectors(n,(1,0,0)) for n in additional_neck_directions]
        additional_neck_y_angles = [get_angle_between_vectors(n,(0,1,0)) for n in additional_neck_directions]
        additional_neck_z_angles = [get_angle_between_vectors(n,(0,0,1)) for n in additional_neck_directions]
        
        #neck shaft angle
        additional_neck_shaft_angles = [get_angle_between_vectors(d,n) for d,n in zip(additional_diaphysis_directions,additional_neck_directions)]
        
        print(f'Femoral Diaphysis axis {np.mean(additional_ct_x_angles):.2f}+/-{np.std(additional_ct_x_angles):.2f} {np.mean(additional_ct_y_angles):.2f}+/-{np.std(additional_ct_y_angles):.2f} {np.mean(additional_ct_z_angles):.2f}+/-{np.std(additional_ct_z_angles):.2f}')
        print(f'Femoral Neck axis {np.mean(additional_neck_x_angles):.2f}+/-{np.std(additional_neck_x_angles):.2f} {np.mean(additional_neck_y_angles):.2f}+/-{np.std(additional_neck_y_angles):.2f} {np.mean(additional_neck_z_angles):.2f}+/-{np.std(additional_neck_z_angles):.2f}')  
        print(f'Neck Shaft Angle {180.0 - np.mean(additional_neck_shaft_angles):.2f} +/- {np.std(additional_neck_shaft_angles):.2f}')      
        print(f'Femoral Head  Radius {femoral_head_radius:.2f} +/- {head_sphere.residue:.2f}')
        print(f'Femoral Head Offset {mean_fho:.2f} +/- {std_fho:.2f}')
    else:
        diaphysis_direction = get_vector_from_points(*l_a)
        ct_x_angle = get_angle_between_vectors(diaphysis_axis,(1,0,0))
        ct_y_angle = get_angle_between_vectors(diaphysis_axis,(0,1,0))
        ct_z_angle = get_angle_between_vectors(diaphysis_axis,(0,0,1))
        
        neck_direction = get_vector_from_points(*get_line_segment(p_n,neck_normal,10)) # type: ignore        
        neck_x_angle = get_angle_between_vectors(neck_normal,(1,0,0))
        neck_y_angle = get_angle_between_vectors(neck_normal,(0,1,0))
        neck_z_angle = get_angle_between_vectors(neck_normal,(0,0,1))

        neck_shaft_angle = get_angle_between_vectors(diaphysis_direction,neck_direction)

        femoral_head_offset = get_distance_to_line_segment(head_sphere.center.tolist(),*l_a)
        femoral_head_offset = femoral_head_offset / 2

        print(f'Femoral Diaphysis axis {ct_x_angle:.2f} {ct_y_angle:.2f} {ct_z_angle:.2f}')
        print(f'Femoral Neck axis {neck_x_angle:.2f} {neck_y_angle:.2f} {neck_z_angle:.2f}')
        print(f'Neck Shaft Angle {180.0 - neck_shaft_angle}')      
        print(f'Femoral Head  Radius {femoral_head_radius:.2f} +/- {head_sphere.residue:.2f}')
        print(f'Femoral Head Offset {femoral_head_offset:.2f}')
    if visualize:
        vedo.show(mesh_obj.c('blue',0.1),
                  head_sphere.wireframe(),
                #   pi_c.opacity(0.5),
                get_arrow_actor(diaphysis_center.GetPosition(),diaphysis_axis).c('green'),
                mesh_obj.clone().cut_with_plane(diaphysis_center.GetPosition(),diaphysis_axis).boundaries().c('green'),
                mesh_obj.clone().cut_with_plane(p_n,neck_normal).boundaries().c('green'),
                get_arrow_actor(p_n,neck_normal).c('green'), 
                axes=1,offscreen=offscreen
                )
        out_filename = Path(nifti_filename).with_suffix('.png')
        vedo.screenshot(str(Path(screenshot_out_dir)/out_filename.name))
        if offscreen:
            vedo.close()

    return {
        'fhr':femoral_head_radius,
        'fna': 180.0 - neck_shaft_angle,
        # 'fnw' : neck_radius,
        'fho': femoral_head_offset,
    }

def test_single():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('nifti_file')
    parser.add_argument('--robust',default=False,action='store_true')
    args = parser.parse_args()

    get_femur_morphometry(args.nifti_file,robust=args.robust,visualize=True,offscreen=False,screenshot=False)


def process_dir():
    import csv

    from tqdm import tqdm

    filenames = list(Path('femur_manual_cut_plane').glob('*.nii.gz'))
    print(f'processing {len(filenames)} files')

    filestream = open(Path('femur_manual_cut_plane/metrics_log')/'metric-log.csv','w')
    filestream_writer = csv.writer(filestream)
    header = ['subject-id', 'FHR(mm)', 'FNA(degrees)','FHO(mm)']
    filestream_writer.writerow(header)

    for f in tqdm(filenames):
        metric_out = get_femur_morphometry(str(f),robust=False,visualize=True,offscreen=True,screenshot=True,screenshot_out_dir='femur_manual_cut_plane/visualize')
        filestream_writer.writerow(
            [ '{:.2f}'.format(item) if type(item) == float or type(item) == np.float64 else item for item in 
                [
                get_nifti_stem(str(f)),
                metric_out['fhr'],
                metric_out['fna'],
                metric_out['fho'],
                ]
            ])
        filestream.flush()
    filestream.close()

if __name__ == '__main__':
    # process_dir()
    test_single()

    