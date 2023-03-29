"""femur morphometry utils"""
from typing import Tuple

import vedo
from xrayto3d_morphometry import (
    cma_es_search_candidate_cut_plane,
    get_angle_between_vectors,
    get_best_solution,
    get_closest_point_from_line,
    get_distance_to_line_segment,
    get_farthest_point_from_line_segment,
    get_line_segment,
    get_mesh_from_segmentation,
    get_points_along_directions,
    get_vector_from_points,
    lerp,
)

femur_label_dict = {"head": 4, "neck": 3, "sub_troc": 2}


def get_neck_shaft_angle(diaphysis_line, neck_line):
    """given line segment representing diaphsis axis and neck axis, evaluate neck shaft angle"""
    diaphysis_normal = get_vector_from_points(*diaphysis_line)
    neck_normal = get_vector_from_points(*neck_line)
    nsa = get_angle_between_vectors(diaphysis_normal, neck_normal)
    return (
        180.0 - nsa if nsa < 90.0 else nsa
    )  # sometimes the normals are pointed in opposite direction, detect this and correct


def get_femoral_head_offset(diaphysis_line, femoral_head: vedo.Sphere):
    """femoral head offset: perpendicular distance of femoral head center (center of rotation) from the diaphysis line"""
    return get_distance_to_line_segment(femoral_head.center.tolist(), *diaphysis_line)


def fit_femoral_head_sphere(p_c, pi_c_normal, mesh_obj) -> vedo.Sphere:
    """fit a sphere on a point cloud representing femoral head

    Args:
        p_c ([float,float,float]]): Point representing farthest point from diaphysis axis
        pi_c_normal ([float,float,float]): vector representing direction of the tangent plane at p_c
        mesh_obj (vedo.Mesh): femur mesh

    Returns:
        vedo.Sphere: Fitted sphere approximating the femoral head
    """
    pc_points = get_points_along_directions(p_c, pi_c_normal, 50, positive_only=True)
    candidate_femoral_head_cuts = [
        mesh_obj.clone().cut_with_plane(p, pi_c_normal).boundaries() for p in pc_points
    ]
    candidate_sphere_points = []
    for cut in candidate_femoral_head_cuts:
        candidate_sphere_points.extend(cut.points().tolist())
    head_sphere: vedo.Sphere = vedo.fit_sphere(candidate_sphere_points)  # type: ignore
    return head_sphere


def get_femur_morphometry(
    nifti_filename, subtrochanter_centroid: Tuple[float, float, float]
):
    """return key:val containing femur morphometry"""
    mesh_obj = get_mesh_from_segmentation(nifti_filename)

    # diaphysis axis
    cma_obj = cma_es_search_candidate_cut_plane(
        mesh_obj, subtrochanter_centroid, (0, 1, 0), verbose=False
    )
    diaphysis_direction = get_best_solution(cma_obj)
    l_a = get_line_segment(subtrochanter_centroid, diaphysis_direction, 400)
    p_c, _ = get_farthest_point_from_line_segment(mesh_obj.points(), *l_a)  # type: ignore
    p_m = get_closest_point_from_line(mesh_obj.center_of_mass(), *l_a)  # type: ignore
    pi_c_normal = get_vector_from_points(p_c, p_m)

    # fit femoral head
    femoral_head = fit_femoral_head_sphere(p_c, pi_c_normal, mesh_obj)

    # fit femoral neck
    l_n = get_vector_from_points(femoral_head.center, p_m)
    p_n = lerp(femoral_head.center.tolist(), p_m, alpha=0.5)
    neck_es = cma_es_search_candidate_cut_plane(mesh_obj, p_n, l_n, verbose=False)
    neck_normal = get_best_solution(neck_es)

    nsa = get_neck_shaft_angle(l_a, get_line_segment(p_n, neck_normal, 10))
    fhr = femoral_head.radius
    fho = get_femoral_head_offset(l_a, femoral_head)
    return {"nsa": nsa, "fhr": fhr, "fho": fho}
