from xrayto3d_morphometry import *
import vedo

def test_geom_ops():
    dist = get_distance_between_points((1,1,1),(1,1,10))
    expected_dist = 9.0
    assert abs(dist-expected_dist) < 1e-5, f'Error:get_distance_between_points {dist:.2f}'

    angle = get_angle_between_vectors((1,0,0),(0,1,0))
    expected_angle = 90
    assert abs(angle-expected_angle) < 1e-5, f'Error:get_angle_between_vectors {angle:.2f}'

    angle = get_angle_between_vectors((1,0,0),(1,1,0))
    expected_angle = 45
    assert abs(angle-expected_angle) < 1e-5, f'Error:get_angle_between_vectors {angle:.2f}'

    angle = get_angle_between_vectors((1,0,0),(-1,0,0))
    expected_angle = 180
    assert abs(angle-expected_angle) < 1e-5, f'Error:get_angle_between_vectors {angle:.2f}'

if __name__ == '__main__':
    test_geom_ops()