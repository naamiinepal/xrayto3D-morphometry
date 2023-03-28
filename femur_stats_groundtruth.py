from pathlib import Path

import numpy as np
import vedo

from xrayto3d_morphometry import (
    extract_volume_surface,
    get_angle_between_vectors,
    get_arrow_actor,
    get_closest_point_from_line,
    get_distance_to_line_segment,
    get_line_segment,
    get_mesh_from_segmentation,
    get_nifti_stem,
    get_points_along_directions,
    get_segmentation_volume,
    get_vector_from_points,
    grid_search_candidate_cut_plane,
    get_farthest_point_from_line_segment,
    lerp,
)


def get_femur_morphometry(
    nifti_filename,
    visualize=True,
    screenshot=True,
    offscreen=True,
    screenshot_out_dir=".",
):
    mesh_obj = get_mesh_from_segmentation(nifti_filename, largest_component=True)

    label_dict = {"head": 4, "neck": 3, "sub_troc": 2}
    subtroc_mesh = extract_volume_surface(
        get_segmentation_volume(nifti_filename, label_dict["sub_troc"])
    )

    # diaphysis axis
    diaphysis_center = vedo.Point(subtroc_mesh.center_of_mass())
    diaphysis_axis, _ = grid_search_candidate_cut_plane(
        mesh_obj, diaphysis_center.GetPosition(), (0, 1, 0)
    )

    # ----use the initial diaphysis axis to traverse through
    additional_diaphysis_center = get_points_along_directions(
        diaphysis_center.GetPosition(), diaphysis_axis
    )
    additional_diaphysis_axis = [
        grid_search_candidate_cut_plane(
            mesh_obj, center, diaphysis_axis, range_min=-0.1, range_max=0.1
        )
        for center in additional_diaphysis_center
    ]

    # heuristics: Cerveri et.al 2010
    # l_a: line representing diaphysis axis
    # p_c: point on the surface at maximum distance from l_a
    # pi_a: plane passing through p_c and l_a
    # p_m: projection of femur center on l_a
    # l_b: line connecting p_c and p_m
    # pi_c: tangent plane at p_c
    l_a = get_line_segment(diaphysis_center.GetPosition(), diaphysis_axis, 400)
    additional_l_a = [
        get_line_segment(center, axis, 400)
        for center, (axis, _) in zip(
            additional_diaphysis_center, additional_diaphysis_axis
        )
    ]
    p_c, p_c_idx = get_farthest_point_from_line_segment(mesh_obj.points(), *l_a)  # type: ignore
    # pi_a = vedo.fit_plane(vedo.Points([*l_a, p_c]))
    p_m = get_closest_point_from_line(mesh_obj.center_of_mass(), *l_a)  # type: ignore
    # l_b = (p_m, p_c)
    pi_c_normal = get_vector_from_points(p_c, p_m)
    # pi_c = vedo.Plane(p_c, pi_c_normal, s=(100, 100))

    # get potential femoral head surface points, then fit femoral head
    pc_points = get_points_along_directions(p_c, pi_c_normal, 50, positive_only=True)
    candidate_femoral_head_cuts = [
        mesh_obj.clone().cut_with_plane(p, pi_c_normal).boundaries() for p in pc_points
    ]
    candidate_sphere_points = []
    for cut in candidate_femoral_head_cuts:
        candidate_sphere_points.extend(cut.points().tolist())
    head_sphere: vedo.Sphere = vedo.fit_sphere(candidate_sphere_points)  # type: ignore

    # fit femoral neck
    l_n = get_vector_from_points(head_sphere.center, p_m)
    p_n = lerp(head_sphere.center.tolist(), p_m, alpha=0.5)
    pi_n, _ = grid_search_candidate_cut_plane(mesh_obj.clone(), p_n, l_n)
    # ----use the initial neck axis to traverse through
    additional_p_n = get_points_along_directions(p_n, pi_n)
    additional_neck_pi_n = np.asarray(
        [
            grid_search_candidate_cut_plane(
                mesh_obj, com, pi_n, range_min=-0.1, range_max=0.1, verbose=False
            )
            for com in additional_p_n
        ]
    )
    min_index = np.argmin(additional_neck_pi_n[:, 1])
    min_p_n = additional_p_n[min_index]
    min_pi_n = additional_neck_pi_n[min_index, 0]
    _, neck_radius, _ = vedo.fit_circle(
        mesh_obj.clone().cut_with_plane(min_p_n, min_pi_n).boundaries().points()
    )

    diaphysis_direction = get_vector_from_points(*l_a)
    ct_x_angle = get_angle_between_vectors(diaphysis_axis, (1, 0, 0))
    ct_y_angle = get_angle_between_vectors(diaphysis_axis, (0, 1, 0))
    ct_z_angle = get_angle_between_vectors(diaphysis_axis, (0, 0, 1))
    additional_diaphysis_directions = [
        get_vector_from_points(*get_line_segment(p, n, 10))
        for p, (n, _) in zip(additional_diaphysis_center, additional_diaphysis_axis)
    ]
    additional_ct_x_angles = [
        get_angle_between_vectors(n, (1, 0, 0)) for n in additional_diaphysis_directions
    ]
    additional_ct_y_angles = [
        get_angle_between_vectors(n, (0, 1, 0)) for n in additional_diaphysis_directions
    ]
    additional_ct_z_angles = [
        get_angle_between_vectors(n, (0, 0, 1)) for n in additional_diaphysis_directions
    ]

    neck_direction = get_vector_from_points(*get_line_segment(p_n, pi_n, 10))  # type: ignore
    additional_neck_directions = [
        get_vector_from_points(*get_line_segment(p, n, 10))
        for p, (n, _) in zip(additional_p_n, additional_neck_pi_n)
    ]
    additional_neck_x_angles = [
        get_angle_between_vectors(n, (1, 0, 0)) for n in additional_neck_directions
    ]
    additional_neck_y_angles = [
        get_angle_between_vectors(n, (0, 1, 0)) for n in additional_neck_directions
    ]
    additional_neck_z_angles = [
        get_angle_between_vectors(n, (0, 0, 1)) for n in additional_neck_directions
    ]
    neck_x_angle = get_angle_between_vectors(pi_n, (1, 0, 0))
    neck_y_angle = get_angle_between_vectors(pi_n, (0, 1, 0))
    neck_z_angle = get_angle_between_vectors(pi_n, (0, 0, 1))

    femoral_neck_angle = get_angle_between_vectors(diaphysis_direction, neck_direction)
    additional_femoral_neck_angles = [
        get_angle_between_vectors(d, n)
        for d, n in zip(additional_diaphysis_directions, additional_neck_directions)
    ]

    femoral_head_radius = (
        head_sphere.radius
    )  # why are we getting radius that is twice as big as it should be ?
    # a normal femoral head is around 16 mm.
    femoral_head_radius = femoral_head_radius / 2  # quick fix: just divide by 2

    femoral_head_offset = get_distance_to_line_segment(
        head_sphere.center.tolist(), *l_a
    )
    femoral_head_offset = femoral_head_offset / 2

    additional_femoral_head_offset = [
        get_distance_to_line_segment(head_sphere.center.tolist(), *l) / 2
        for l in additional_l_a
    ]
    mean_fho = np.mean(additional_femoral_head_offset)
    std_fho = np.std(additional_femoral_head_offset)

    print(
        f"Femoral Diaphysis axis {ct_x_angle:.2f} {ct_y_angle:.2f} {ct_z_angle:.2f} {np.mean(additional_ct_x_angles):.2f}+/-{np.std(additional_ct_x_angles):.2f} {np.mean(additional_ct_y_angles):.2f}+/-{np.std(additional_ct_y_angles):.2f} {np.mean(additional_ct_z_angles):.2f}+/-{np.std(additional_ct_z_angles):.2f}"
    )

    print(
        f"Femoral Neck axis {neck_x_angle:.2f} {neck_y_angle:.2f} {neck_z_angle:.2f} {np.mean(additional_neck_x_angles):.2f}+/-{np.std(additional_neck_x_angles):.2f} {np.mean(additional_neck_y_angles):.2f}+/-{np.std(additional_neck_y_angles):.2f} {np.mean(additional_neck_z_angles):.2f}+/-{np.std(additional_neck_z_angles):.2f}"
    )

    print(
        f"Femoral Head  Radius {femoral_head_radius:.2f} +/- {head_sphere.residue:.2f}"
    )
    print(
        f"Femoral Neck Angle {180.0 - femoral_neck_angle:.2f} {180.0 - np.mean(additional_femoral_neck_angles):.2f} +/- {np.std(additional_femoral_neck_angles):.2f}"
    )
    print(f"Femoral (min)Neck Width {neck_radius:.2f}")
    print(f"Femoral Head Offset {mean_fho:.2f} +/- {std_fho:.2f}")

    if visualize:
        vedo.show(
            mesh_obj.c("blue", 0.1),
            diaphysis_center,
            get_arrow_actor(diaphysis_center.GetPosition(), diaphysis_axis),
            head_sphere.c("red", 0.3).wireframe(),
            vedo.Point(p_n),
            mesh_obj.clone().cut_with_plane(p_n, pi_n).boundaries(),
            mesh_obj.clone().cut_with_plane(min_p_n, min_pi_n).boundaries().c("blue"),
            axes=1,
            offscreen=offscreen,
        )
        if screenshot:
            out_filename = Path(nifti_filename).with_suffix(".png")
            vedo.screenshot(str(Path(screenshot_out_dir) / out_filename.name))
        if offscreen:
            vedo.close()
    return {
        "fhr_mean": femoral_head_radius,
        "fhr_std": head_sphere.residue,
        "fna_mean": 180.0 - np.mean(additional_femoral_neck_angles),
        "fna_std": np.std(additional_femoral_neck_angles),
        "fnw": neck_radius,
        "fho_mean": mean_fho,
        "fho_std": std_fho,
    }


def test_single():
    """test single example"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nifti_file")
    parser.add_argument("--visualize", default=False, action="store_true")
    parser.add_argument("--offscreen", default=False, action="store_true")
    parser.add_argument("--screenshot", default=False, action="store_true")
    args = parser.parse_args()

    get_femur_morphometry(
        args.nifti_file,
        visualize=args.visualize,
        offscreen=args.offscreen,
        screenshot=args.screenshot,
    )


def process_dir():
    """process all segmentation in a directory"""
    import csv

    from tqdm import tqdm

    filenames = list(Path("femur_manual_cut_plane").glob("*.nii.gz"))
    print(f"processing {len(filenames)} files")

    filestream = open(
        Path("femur_manual_cut_plane/metrics_log") / "metric-log.csv", "w"
    )
    filestream_writer = csv.writer(filestream)
    header = [
        "subject-id",
        "FHR(mm)",
        "FNA(degrees)",
        "FNW(mm)",
        "FHO(mm)",
        "FHR(std)",
        "FNA(std)",
        "FHO(std)",
    ]
    filestream_writer.writerow(header)

    for f in tqdm(filenames):
        metric_out = get_femur_morphometry(
            str(f),
            visualize=True,
            offscreen=True,
            screenshot=True,
            screenshot_out_dir="femur_manual_cut_plane/visualize",
        )
        filestream_writer.writerow(
            [
                "{:.2f}".format(item)
                if type(item) == float or type(item) == np.float64
                else item
                for item in [
                    get_nifti_stem(str(f)),
                    metric_out["fhr_mean"],
                    metric_out["fna_mean"],
                    metric_out["fnw"],
                    metric_out["fho_mean"],
                    metric_out["fhr_std"],
                    metric_out["fna_std"],
                    metric_out["fho_std"],
                ]
            ]
        )
        filestream.flush()
    filestream.close()


if __name__ == "__main__":
    process_dir()
