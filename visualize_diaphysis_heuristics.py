from xrayto3d_morphometry import *
import vedo

def process_femur_heuristics(nifti_filename):
    mesh_obj = get_mesh_from_segmentation(nifti_filename,largest_component=True)

    label_dict = {'head':4,'neck':3,'sub_troc':2}
    subtroc_mesh = extract_volume_surface(get_segmentation_volume(nifti_filename,label_dict['sub_troc']))
    # diaphysis axis
    diaphysis_com = vedo.Point(subtroc_mesh.center_of_mass())
    initial_diaphysis_cpn,csa = grid_search_candidate_cut_plane(mesh_obj,diaphysis_com.GetPosition(),(0,1,0),verbose=True)

    # heuristics: Cerveri et.al. 2010
    #i_a: line representing diaphysis axis
    l_a = get_line_segment(diaphysis_com.GetPosition(),initial_diaphysis_cpn,50)
    #p_c: point on the surface at maximum distance from i_a
    p_c,p_c_idx= get_farthest_point_from_line_segment(mesh_obj.points(),*l_a)
    # fit plane pi_a passing through p_c and i_a
    pi_a = vedo.fit_plane(vedo.Points([*l_a,p_c]))
    mesh_com = mesh_obj.center_of_mass()
    #project p_m on l_a
    p_m = get_closest_point_from_line(mesh_com,*l_a)
    #l_b: line connection p_c and projection of mesh center on l_a
    l_b = (p_m,p_c)
    #pi_c: tangent plane at p_c
    pi_c_normal = get_vector_from_points(p_c,p_m)
    pi_c = vedo.Plane(p_c,pi_c_normal,s=(40,40))
    pc_points = get_points_along_directions(p_c,pi_c_normal,50,positive_only=True)

    # fit femoral head
    # pick cross sections 
    candidate_femoral_head_cuts = [ mesh_obj.clone().cut_with_plane(p,pi_c_normal).boundaries() for p in pc_points]
    largest_cross_section = sys.float_info.min
    femoral_head_cuts = []
    sphere_points = []
    for hc in candidate_femoral_head_cuts:
        csa = hc.clone().triangulate().area()
        if csa > largest_cross_section:
            largest_cross_section = csa
            femoral_head_cuts.append(hc)
            sphere_points.extend(hc.points().tolist())
        else:
            break
    try:
        head_sph:vedo.Sphere = vedo.fit_sphere(sphere_points)
    except ValueError as e:
        print(nifti_filename,'ERROR',e)
        
    # fit femoral neck
    # initial femoral neck axis
    init_fna_line = vedo.Line(head_sph.center,p_m)
    init_fna = get_vector_from_points(head_sph.center,p_m)
    init_neck_center = lerp(head_sph.center.tolist(),p_m,alpha=0.5)
    candidate_neck_cut_plane,csa = grid_search_candidate_cut_plane(mesh_obj.clone(),init_neck_center,init_fna)
    vedo.show(
        mesh_obj.opacity(0.1),
        diaphysis_com,
        subtroc_mesh.opacity(0.3),
        get_arrow_actor(diaphysis_com.GetPosition(),initial_diaphysis_cpn).opacity(0.3),
        vedo.Line(l_a),
        vedo.Point(p_c).c('blue'),
        vedo.Point(mesh_com).c('blue'),
        # pi_a,
        vedo.Point(mesh_com),
        vedo.Point(p_m).c('blue'),
        vedo.Line(l_b),
        # pi_c,
        *femoral_head_cuts,
        head_sph.opacity(0.3),
        init_fna_line,
        vedo.Point(init_neck_center).c('blue'),
        mesh_obj.clone().cut_with_plane(init_neck_center,candidate_neck_cut_plane).boundaries(),
        resetcam=False,
        camera=get_oriented_camera(mesh_obj,2,-400),
        zoom=1.5
    )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('nifti_file')
    args = parser.parse_args()

    process_femur_heuristics(args.nifti_file)