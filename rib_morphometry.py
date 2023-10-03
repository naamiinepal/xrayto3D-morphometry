import vedo
from xrayto3d_morphometry import (
    get_mesh_from_segmentation,
    move_to_origin,
    get_oriented_camera,
    get_distance_between_points,
)
import json
import numpy as np


def get_coords(endpoint_path):
    with open(endpoint_path, "r") as f:
        endpoint_json = json.load(f)
        control_points = endpoint_json["markups"][0]["controlPoints"]
        points = [p["position"] for p in control_points]
    return points


def arc_length(arrPoints):
    previous = arrPoints[0]
    dist = 0
    for i, point in enumerate(arrPoints):
        # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
        dist += np.linalg.norm(np.asarray(point) - np.asarray(previous))
        previous = point
    return dist


sample_file = (
    "2D-3D-Reconstruction-Datasets/rib-centerline/s0046/s0046_rib_msk_gt_1.nii.gz"
)

rib_mesh = get_mesh_from_segmentation(
    sample_file, largest_component=False, reorient=False
)
move_to_origin(rib_mesh)

# path templates
endpoint_path = "2D-3D-Reconstruction-Datasets/rib-centerline/s0046/Endpoints.mrk.json"

endpoint_path_template = (
    "2D-3D-Reconstruction-Datasets/rib-centerline/s0046/Endpoints_{i}.mrk.json"
)

center_line_1_path = (
    "2D-3D-Reconstruction-Datasets/rib-centerline/s0046/Centerline curve (0).mrk.json"
)

center_line_template = "2D-3D-Reconstruction-Datasets/rib-centerline/s0046/Centerline curve_{i} (0).mrk.json"

# get all paths into a list
endpoint_paths = [
    endpoint_path,
]
for i in range(1, 24):
    endpoint_paths.append(endpoint_path_template.format(i=i))

center_line_paths = [
    center_line_1_path,
]
for i in range(1, 24):
    center_line_paths.append(center_line_template.format(i=i))

# get rib-midline length
for p in center_line_paths:
    print("arc length", arc_length(get_coords(p)) * 2.5)

# get chordal length
for p in endpoint_paths:
    p1, p2 = get_coords(p)
    print("chord length", get_distance_between_points(p1, p2) * 2.5)

# get rib area
rib_surfaces = []
for p in center_line_paths:
    rib_surface = vedo.delaunay2d(vedo.Points(get_coords(p)))
    rib_surfaces.append(rib_surface)
    print("rib area", rib_surface.area() * 2.5 * 2.5)

# get best-fit plane orientation


# for visualization
endpoints = []
endpoints.extend(get_coords(endpoint_path))
for i in range(1, 24):
    endpoints.extend(get_coords(endpoint_path_template.format(i=i)))

center_lines = []
center_lines.extend(get_coords(center_line_1_path))
for i in range(1, 24):
    center_lines_pts = get_coords(center_line_template.format(i=i))
    center_lines.extend(center_lines_pts)

frontview_cam = get_oriented_camera(rib_mesh, axis=2, camera_dist=600)
frontview_cam["viewup"] = (0, -1, 0)
vedo.show(
    # rib_mesh,
    # rib_bounds,
    vedo.Points(endpoints, r=10),
    vedo.Points(center_lines, c=(1.0, 0, 0)),
    *rib_surfaces,
    axes=1,
    camera=frontview_cam,
    resetcam=False,
)
