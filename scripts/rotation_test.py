import vedo
import numpy as np
from xrayto3d_morphometry import get_mesh_from_segmentation, move_to_origin, get_vector_from_points, lerp

def get_app_plane_rotation_matrix(
    pt_p1_coord, pt_p2_coord, asis_p1_coord, asis_p2_coord
):
    """return 3x3 rotation matrix that defines the axes of the APP plane
    defined by the ASIS and Pubic Symphysis"""
    x_direction = get_vector_from_points(pt_p1_coord, pt_p2_coord)
    pt_mid = lerp(pt_p1_coord, pt_p2_coord, 0.5)
    app_points = vedo.Points([asis_p1_coord, asis_p2_coord, pt_mid])
    app_plane = vedo.fit_plane(app_points)
    normal = -app_plane.normal
    transformation_matrix = np.array([x_direction, np.cross(normal, x_direction), normal])
    return transformation_matrix

landmarks = [[-79.967545,  21.783907,  50.0734 ],
             [85.032455, 11.783907, 53.0734 ],
             [-16.967546, -39.21609 ,  50.0734],
             [13.032453, -44.21609 ,  50.0734]]
asis_plane = vedo.fit_plane(vedo.Points(landmarks))

nifti_filename = 'test_data/s0014_hip_msk_pred.nii.gz'
mesh_obj = get_mesh_from_segmentation(nifti_filename)
mesh_obj.rotate_x(180, around=mesh_obj.center_of_mass())

move_to_origin(mesh_obj)
T = get_app_plane_rotation_matrix(landmarks[2],landmarks[3],landmarks[0],landmarks[1])
print(T)
tx_mesh_obj = mesh_obj.clone(deep=True, transformed=True)
tx_mesh_obj.apply_transform(T, concatenate=True)

vedo.show(mesh_obj.c('white',0.4), tx_mesh_obj.c('green',0.4), vedo.Points(landmarks,c='blue',r=24), asis_plane.opacity(0.2), axes=1)
