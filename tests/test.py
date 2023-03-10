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

if __name__ == '__main__':
    test_move_to_origin()