from xrayto3d_morphometry import *

def main():
    sample_fp = 'test_data/s0000_femur_left_msk_detailed_4class.vtk'
    mesh_obj = read_mesh(sample_fp)
    # diaphysis axis
    diaphysis_com = vedo.Point(get_pointcloud_from_mesh(mesh_obj,label=4).center_of_mass())
    initial_diaphysis_cpn,csa = grid_search_candidate_cut_plane(mesh_obj,diaphysis_com.GetPosition(),(0,0,1),verbose=False)
    ##----use the initial diaphysis axis to traverse through
    additional_diaphysis_com = get_points_along_directions(diaphysis_com.GetPosition(),initial_diaphysis_cpn)
    additional_diaphysis_cpn = [grid_search_candidate_cut_plane(mesh_obj,com,initial_diaphysis_cpn,range_min=-0.1,range_max=0.1) for com in additional_diaphysis_com]
    
    # neck axis
    neck_com = vedo.Point(get_pointcloud_from_mesh(mesh_obj,label=2).center_of_mass())
    initial_neck_cpn,csa = grid_search_candidate_cut_plane(mesh_obj,neck_com.GetPosition(),(1,0,0))
    ##----use the initial neck axis to traverse through
    additional_neck_com = get_points_along_directions(neck_com.GetPosition(),initial_neck_cpn)
    additional_neck_cpn = [grid_search_candidate_cut_plane(mesh_obj,com,initial_neck_cpn,range_min=-0.1,range_max=0.1) for com in additional_diaphysis_com]

    # head sphere
    head_com = vedo.Point(get_pointcloud_from_mesh(mesh_obj,label=1).center_of_mass())
    head_sph = vedo.fit_sphere(get_pointcloud_from_mesh(mesh_obj,label=1))

    vedo.show(mesh_obj.opacity(0.3),
              diaphysis_com,
              mesh_obj.clone().cut_with_plane(diaphysis_com.GetPosition(),initial_diaphysis_cpn).boundaries(),
              get_arrow_actor(diaphysis_com.GetPosition(),initial_diaphysis_cpn).opacity(0.3),
              vedo.Points(additional_diaphysis_com,r=10),
              *[get_arrow_actor(c,n).opacity(0.1) for c,(n,_) in zip(additional_diaphysis_com,additional_diaphysis_cpn)],
              *[mesh_obj.clone().cut_with_plane(c,n).boundaries() for c,(n,_) in zip(additional_diaphysis_com,additional_diaphysis_cpn)],
              neck_com,
              mesh_obj.clone().cut_with_plane(neck_com.GetPosition(),initial_neck_cpn).boundaries(),
              get_arrow_actor(neck_com.GetPosition(),initial_neck_cpn).opacity(0.3),
              vedo.Points(additional_neck_com,r=10),
              *[get_arrow_actor(c,n).opacity(0.1) for c,(n,_) in zip(additional_neck_com,additional_neck_cpn)],
              *[mesh_obj.clone().cut_with_plane(c,n).boundaries() for c,(n,_) in zip(additional_neck_com,additional_neck_cpn)],
              head_com,
              head_sph.opacity(0.3),
              resetcam=False,camera=get_oriented_camera(mesh_obj,1,-400),axes=1)

if __name__ == '__main__':
    main()