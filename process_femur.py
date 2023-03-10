from xrayto3d_morphometry import *

def main():
    sample_fp = 'test_data/s0000_femur_left_msk_detailed_4class.vtk'
    mesh_obj = read_mesh(sample_fp)
    neck_com = vedo.Point(get_pointcloud_from_mesh(mesh_obj,label=4).center_of_mass())
    cpn,csa = grid_search_candidate_cut_plane(mesh_obj,neck_com.GetPosition(),(0,0,1),verbose=True)
    vedo.show(mesh_obj.opacity(0.3),
              neck_com,
              mesh_obj.clone().cutWithPlane(neck_com.GetPosition(),cpn).boundaries(),
              get_arrow_actor(neck_com.GetPosition(),cpn),
              resetcam=False,camera=get_oriented_camera(mesh_obj,1,-400),axes=1)

if __name__ == '__main__':
    main()