"""write pelvic landmarks into pelvic_landmarks.csv"""
import argparse
import csv
import os
import functools
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Any
import vedo
from xrayto3d_morphometry import (
    get_nifti_stem,
    get_mesh_from_segmentation,
    move_to_origin,
    align_along_principal_axes,
    get_oriented_camera,
)
from pelvic_landmark_utils import get_quadrant_meshes, get_maximal_pelvic_points

landmarks = [
    "ASIS_L",
    "ASIS_R",
    "PT_L",
    "PT_R",
]


def write_metric_log_header(filepath, filename="pelvic_landmarks.csv"):
    """write output log header"""
    outdir = Path(f"{filepath}/metrics_log")
    outdir.mkdir(exist_ok=True)
    with open(outdir / f"{filename}", "w", encoding="utf-8") as f:
        filestream_writer = csv.writer(f)
        header = ["ASIS_L,ASIS_R,PT_L,PT_R"]
        filestream_writer.writerow(header)


def get_landmark_formatted_row(nifti_file, landmarks):
    """output formatted string containing comma-separated landmarks"""
    return f"{get_nifti_stem(str(nifti_file))}"


def pelvic_landmark_helper(filepath, log_dir, filename):
    """helper func"""
    if isinstance(filepath, Path):
        filepath = str(filepath)
    landmarks = ...
    with open(f"{log_dir}/metrics_log/{filename}", "a", encoding="utf-8") as f:
        f.write(get_landmark_formatted_row(filepath, landmarks))


def visualize_mesh(nifti_filename, screenshot=False, screenshot_outdir="."):
    mesh = get_mesh_from_segmentation(nifti_filename)
    move_to_origin(mesh)
    aligned_mesh, T = align_along_principal_axes(mesh)
    bl, tl, br, tr = get_quadrant_meshes(mesh)
    mwp_p1_idx, mwp_p2_idx = get_maximal_pelvic_points(aligned_mesh)
    cam = get_oriented_camera(aligned_mesh, 2, 400)
    vedo.show(
        *get_quadrant_meshes(aligned_mesh),
        vedo.Point(aligned_mesh.points()[mwp_p1_idx], c="red"),
        vedo.Point(aligned_mesh.points()[mwp_p2_idx], c="green"),
        resetcam=False,
        camera=cam,
        axes=1,
        offscreen=screenshot,
    )
    if screenshot:
        out_filename = (
            Path(screenshot_outdir) / Path(nifti_filename).with_suffix(".png").name
        )
        vedo.screenshot(out_filename)
        vedo.close()


def test_single_example():
    """test stub"""
    sample_nifti = "test_data/s0014_hip_msk_pred.nii.gz"
    visualize_mesh(sample_nifti, screenshot=True, screenshot_outdir="visualizations")
    pass


def process_dir():
    """process all files in a dir"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--predicted", default=False, action="store_true")
    args = parser.parse_args()
    # write ouput file header
    write_metric_log_header(args.dir)
    if args.predicted:
        suffix = "_pred.nii.gz"
    else:
        suffix = "_gt.nii.gz"

    filenames = sorted(list(Path(args.dir).glob(f"*{suffix}")))
    print(f"processing {len(filenames)} files")
    worker_fn = functools.partial(
        pelvic_landmark_helper, log_dir=args.dir, filename=args.log_filename
    )
    run_multiple_processes(worker_fn, filenames)


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
    # process_dir()
    test_single_example()
