"""write pelvic landmarks into pelvic_landmarks.csv"""
import argparse
import csv
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable

import numpy as np
import vedo

from pelvic_landmark_utils import (
    get_asis_estimate,
    get_maximal_pelvic_points,
    get_quadrant_meshes,
    get_transverse_plane_height,
)
from xrayto3d_morphometry import (
    align_along_principal_axes,
    get_mesh_from_segmentation,
    get_nifti_stem,
    get_oriented_camera,
    lerp,
    move_to_origin,
)

landmarks = [
    "ASIS_L",
    "ASIS_R",
    "PT_L",
    "PT_R",
]
np.set_printoptions(precision=3, suppress=True)


def write_metric_log_header(filepath, filename="pelvic_landmarks.csv"):
    """write output log header"""
    outdir = Path(f"{filepath}/")
    outdir.mkdir(exist_ok=True)
    with open(outdir / f"{filename}", "w", encoding="utf-8") as f:
        filestream_writer = csv.writer(f)
        header = ["filename", "ASIS_L", "ASIS_R", "PT_L", "PT_R"]
        filestream_writer.writerow(header)


def get_landmark_formatted_row(nifti_file, landmarks):
    """output formatted string containing comma-separated landmarks"""
    return f"{get_nifti_stem(str(nifti_file))},{landmarks['ASIS_L'][0]},{landmarks['ASIS_L'][1]},{landmarks['ASIS_L'][2]},{landmarks['ASIS_R'][0]},{landmarks['ASIS_R'][1]},{landmarks['ASIS_R'][2]},{landmarks['PT_L'][0]},{landmarks['PT_L'][1]},{landmarks['PT_L'][2]},{landmarks['PT_R'][0]},{landmarks['PT_R'][1]},{landmarks['PT_R'][2]}\n"


def pelvic_landmark_helper(nifti_filepath, log_dir, log_filename):
    """helper func"""
    if isinstance(nifti_filepath, Path):
        nifti_filepath = str(nifti_filepath)
    landmarks = get_landmarks(
        nifti_filepath,
        screenshot=True,
        debug=True,
        screenshot_outdir=f"{log_dir}/visualizations",
    )
    with open(f"{log_dir}/{log_filename}", "a", encoding="utf-8") as f:
        f.write(get_landmark_formatted_row(nifti_filepath, landmarks))


def get_landmarks(nifti_filename, screenshot=False, screenshot_outdir=".", debug=False):
    """return landmarks as dict, also take snapshot if required of landmarks"""
    mesh = get_mesh_from_segmentation(nifti_filename)
    move_to_origin(mesh)
    aligned_mesh, T = align_along_principal_axes(mesh)

    # Orientation AFTER principal axis transformation has to be:
    #     with respect to patient(which affects left-to-right orientation) assume segmentation volume is in PIR
    #      ______________________________________________________
    #     |    Axes    |      X      |      Y      |      Z      |
    #     |  Positive  |    Left     |    Inferior |  Posterior  |
    #     |  Negative  |    Right    |    Superior |  Anterior   |
    #     |______________________________________________________|

    mwp_p1_idx, mwp_p2_idx = get_maximal_pelvic_points(aligned_mesh)
    mwp_midpoint = lerp(
        aligned_mesh.points()[mwp_p1_idx], aligned_mesh.points()[mwp_p2_idx], alpha=0.5
    )
    tph_intercept, dlc, drc = get_transverse_plane_height(
        aligned_mesh, mwp_midpoint, alpha=0.6, verbose=True
    )
    tl, bl, tr, br = get_quadrant_meshes(aligned_mesh, (0, tph_intercept, 0))

    i = 0
    init_pt_p1_idx, init_pt_p2_idx, init_asis_p1_idx, init_asis_p2_idx = (
        None,
        None,
        None,
        None,
    )
    pt_p1_idx, pt_p2_idx, asis_p1_idx, asis_p2_idx, T = get_asis_estimate(
        aligned_mesh, (0, tph_intercept, 0), verbose=True
    )
    init_pt_p1_idx, init_pt_p2_idx, init_asis_p1_idx, init_asis_p2_idx = (
        pt_p1_idx,
        pt_p2_idx,
        asis_p1_idx,
        asis_p2_idx,
    )
    while True:
        try:
            aligned_mesh.apply_transform(T, concatenate=True)

            pt_p1_idx, pt_p2_idx, asis_p1_idx, asis_p2_idx, new_T = get_asis_estimate(
                aligned_mesh, (0, tph_intercept, 0), verbose=True
            )
            init_pt_p1_idx, init_pt_p2_idx, init_asis_p1_idx, init_asis_p2_idx = (
                pt_p1_idx,
                pt_p2_idx,
                asis_p1_idx,
                asis_p2_idx,
            )
            if np.allclose(new_T, np.eye(3), atol=0.02) or np.allclose(
                T, new_T, atol=0.02
            ) or i > 50:
                break
            else:
                i += 1
                T = new_T
                if debug:
                    cam = get_oriented_camera(aligned_mesh, 2, 400)
                    vedo.show(
                        aligned_mesh.opacity(0.3),
                        bl.c("red"),
                        tr.c("green"),
                        vedo.Point(aligned_mesh.points()[mwp_p1_idx], c="red"),
                        vedo.Point(aligned_mesh.points()[mwp_p2_idx], c="green"),
                        vedo.Point(mwp_midpoint, c="black"),
                        vedo.Point(dlc, c="white"),
                        vedo.Point(drc, c="white"),
                        vedo.Plane((0, tph_intercept, 0), (0, 1, 0), (100, 100)),
                        vedo.Point(aligned_mesh.points()[pt_p1_idx], c="white"),
                        vedo.Point(aligned_mesh.points()[pt_p2_idx], c="white"),
                        vedo.Point(aligned_mesh.points()[asis_p1_idx], c="white"),
                        vedo.Point(aligned_mesh.points()[asis_p2_idx], c="white"),
                        resetcam=False,
                        camera=cam,
                        axes=1,
                        offscreen=screenshot,
                    )
                    if screenshot:
                        Path(screenshot_outdir).mkdir(exist_ok=True)
                        out_filename = (
                            Path(screenshot_outdir)
                            / f'{i}_{Path(nifti_filename).with_suffix(".png").name}'
                        )
                        vedo.screenshot(out_filename)
                        vedo.close()
        except ValueError as e:
            # fallback if no convergence
            pt_p1_idx, pt_p2_idx, asis_p1_idx, asis_p2_idx = (
                init_pt_p1_idx,
                init_pt_p2_idx,
                init_asis_p1_idx,
                init_asis_p2_idx,
            )
            break

    landmarks = {
        "ASIS_L": mesh.points()[asis_p1_idx],
        "ASIS_R": mesh.points()[asis_p2_idx],
        "PT_L": mesh.points()[pt_p1_idx],
        "PT_R": mesh.points()[pt_p2_idx],
    }

    return landmarks


def test_single_example():
    """test stub"""
    parser = argparse.ArgumentParser()
    parser.add_argument("file")

    args = parser.parse_args()
    if args.file:
        sample_nifti = args.file
    else:
        sample_nifti = "test_data/s0014_hip_msk_pred.nii.gz"
    get_landmarks(
        sample_nifti, screenshot=True, screenshot_outdir="visualizations", debug=True
    )
    pass


def process_dir():
    """process all files in a dir"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--predicted", default=False, action="store_true")
    parser.add_argument("--log_filename", type=str)
    args = parser.parse_args()
    # write ouput file header
    write_metric_log_header(args.dir, filename=args.log_filename)
    if args.predicted:
        suffix = "_pred.nii.gz"
    else:
        suffix = "_gt.nii.gz"

    filenames = sorted(list(Path(args.dir).glob(f"*{suffix}")))
    print(f"processing {len(filenames)} files")
    for filename in filenames:
        print(filename)
        pelvic_landmark_helper(
            filename, log_dir=args.dir, log_filename=args.log_filename
        )


def run_multiple_processes(f_n: Callable, data: Any, num_workers=os.cpu_count()):
    """multiprocessing with output to file
    https://stackoverflow.com/questions/13446445/python-multiprocessing-safely-writing-to-a-file
    https://medium.com/geekculture/python-multiprocessing-with-output-to-file-a6748a27ed41
    """
    pool = Pool(processes=num_workers)
    jobs = []
    for item in data:
        job = pool.apply_async(f_n, (item,))
        jobs.append(job)
    for job in jobs:
        job.get()
    pool.close()
    pool.join()


if __name__ == "__main__":
    process_dir()
    # test_single_example()
