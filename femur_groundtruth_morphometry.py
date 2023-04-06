import argparse
import csv
import os
import functools
from multiprocessing import Pool
from pathlib import Path

from xrayto3d_morphometry import (
    get_femur_morphometry,
    get_nifti_stem,
    get_subtrochanter_center,
    seg_contain_subtrochanter,
)

FEMUR_MANUAL_CUT_PLANE_DIR = (
    "2D-3D-Reconstruction-Datasets/morphometry/femur_manual_cut_plane"
)


def get_metric_formatted_row(nifti_file, metrics_dict):
    """output formatted string containing comma-separated metrics"""
    return f'{get_nifti_stem(str(nifti_file))},{metrics_dict["fhr"]:.2f},{metrics_dict["nsa"]:.2f},{metrics_dict["fho"]:.2f}\n'


def femur_morphometry_helper(filepath, log_dir=FEMUR_MANUAL_CUT_PLANE_DIR):
    """helper func"""
    if isinstance(filepath, Path):
        filepath = str(filepath)
    if not seg_contain_subtrochanter(filepath):
        return
    metrics_dict = get_femur_morphometry(filepath, get_subtrochanter_center(filepath))
    print(get_nifti_stem(filepath), metrics_dict)
    with open(f"{log_dir}/metrics_log/femur_clinical.csv", "a") as f:
        f.write(get_metric_formatted_row(filepath, metrics_dict))


def write_metric_log_header(filepath):
    """write output log header"""
    outdir = Path(f"{filepath}/metrics_log")
    outdir.mkdir(exist_ok=True)
    with open(outdir / "femur_clinical.csv", "w") as filestream:
        filestream_writer = csv.writer(filestream)
        header = ["subject-id", "FHR(mm)", "FNA(degrees)", "FHO(mm)"]
        filestream_writer.writerow(header)


def reconstructed_femur_morphometry_helper(
    predicted_filepath, log_dir=FEMUR_MANUAL_CUT_PLANE_DIR, verbose=False
):
    """obtain metrics for reconstructed femur"""
    if isinstance(predicted_filepath, Path):
        predicted_filepath = str(predicted_filepath)

    file_prefix = get_nifti_stem(predicted_filepath)[:-5]
    manual_localization_file = list(
        Path(FEMUR_MANUAL_CUT_PLANE_DIR).glob(f"{file_prefix}_gt.nii.gz")
    )
    if len(manual_localization_file) != 1:
        print(
            f"Expected 1 but got {len(manual_localization_file)} files with prefix {file_prefix}"
        )
        return
    gt_filepath = str(manual_localization_file[0])
    if not seg_contain_subtrochanter(gt_filepath):
        return

    metrics_dict = get_femur_morphometry(
        predicted_filepath, get_subtrochanter_center(gt_filepath)
    )
    with open(f"{log_dir}/metrics_log/femur_clinical.csv", "a") as f:
        f.write(get_metric_formatted_row(predicted_filepath, metrics_dict))
    if verbose:
        print(gt_filepath)
        print(get_nifti_stem(predicted_filepath), metrics_dict)


def process_dir():
    """process all files in a dir"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default=FEMUR_MANUAL_CUT_PLANE_DIR,
    )
    parser.add_argument("--predicted", default=False, action="store_true")
    args = parser.parse_args()

    # create output file header
    write_metric_log_header(args.dir)
    if args.predicted:
        suffix = "_pred.nii.gz"
        worker_fn = functools.partial(
            reconstructed_femur_morphometry_helper, log_dir=args.dir, verbose=False
        )
    else:
        suffix = "_gt.nii.gz"
        worker_fn = femur_morphometry_helper
    filenames = list(Path(args.dir).glob(f"*{suffix}"))
    print(f"processing {len(filenames)} files")

    num_workers = os.cpu_count()
    pool = Pool(processes=num_workers)
    jobs = []
    for item in sorted(filenames):
        job = pool.apply_async(worker_fn, (item,))
        jobs.append(job)
    for job in jobs:
        job.get()
    pool.close()
    pool.join()


if __name__ == "__main__":
    process_dir()
