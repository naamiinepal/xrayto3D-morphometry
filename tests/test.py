from xrayto3d_morphometry import *
import vedo

def read_sample_mesh():
    sample_mesh_fp = 'test_data/s0000_femur_left_msk.vtk'
    sample_mesh = read_mesh(sample_mesh_fp)
    return sample_mesh

def test_move_to_origin():
    sample_mesh = read_sample_mesh()
    move_to_origin(sample_mesh)
    vedo.show(sample_mesh,axes=1)

def test_get_furthest_point():
    sample_mesh = read_sample_mesh()
    p0,p0_idx = get_furthest_point_along_axis(sample_mesh.points(),0)
    p1,p1_idx = get_furthest_point_along_axis(sample_mesh.points(),1)
    p2,p2_idx = get_furthest_point_along_axis(sample_mesh.points(),2)
    p2_neg,p2_neg_idx = get_furthest_point_along_axis(sample_mesh.points(),2,True)
    vedo.show(sample_mesh,
              vedo.Point(p0),
              vedo.Point(p1),
              vedo.Point(p2),
              vedo.Point(p2_neg),
              axes=1)

def test_get_principal_axis():
    sample_mesh = vedo.Box((0,0,0),10,20,30).rotate_x(45)
    axes_matrix, ellipsoid = get_principal_axis(sample_mesh)
    axes_actor = get_custom_axes_from_ellipsoid(ellipsoid)
    vedo.show(axes_actor,sample_mesh.opacity(0.5))

if __name__ == '__main__':
    # test_move_to_origin()
    # test_get_furthest_point()
    test_get_principal_axis()