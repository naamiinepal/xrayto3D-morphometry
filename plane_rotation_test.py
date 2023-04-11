import vedo
import numpy as np
from xrayto3d_morphometry import get_vector_from_points, lerp, get_direction_axes

landmarks = [
    [14.558963, -47.695496, -42.968834],
    [-15.73021, -50.445164, -42.89637],
    [81.96809, 7.82188, -50.95824],
    [-83.269356, 9.466893, -56.238384],
]
points = vedo.Points(landmarks)
n = [-0.02960122, 0.18161167, 0.98292464]
p = vedo.shapes.Plane(points.center_of_mass(), n, s=[200, 200])
pt_mid = lerp(landmarks[0], landmarks[1], 0.5)
# Temporary rotation matrix
x_direction = get_vector_from_points(landmarks[1], landmarks[0])
T = np.array([x_direction, np.cross(n, x_direction), n])
print(T)
plane_axes = get_direction_axes(pt_mid, x_direction, np.cross(n, x_direction), n)
p_aligned = p.clone(transformed=True)
p_aligned.apply_transform(T, concatenate=True)
aligned_points = points.clone().apply_transform(T, concatenate=True)
new_x = get_vector_from_points(aligned_points.points()[1], aligned_points.points()[0])
print(new_x)
vedo.show(p, p_aligned.c("red"), points, aligned_points, *plane_axes, axes=1)
