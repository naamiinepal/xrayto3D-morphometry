"""evaluate method variability"""
from pathlib import Path
import csv
from xrayto3d_morphometry import (
    get_femur_morphometry,
    get_nifti_stem,
    seg_contain_subtrochanter,
    get_subtrochanter_center,
)

FEMUR_MANUAL_CUT_PLANE_DIR = (
    "2D-3D-Reconstruction-Datasets/morphometry/femur_manual_cut_plane"
)
header = [
    "subject-id",
    "FHR(mm)",
    "NSA(degrees)",
    "FO(mm)",
    "FHC_x(mm)",
    "FHC_y(mm)",
    "FHC_z(mm)",
    "FDA_x(deg)",
    "FDA_y(deg)",
    "FDA_z(deg)",
    "FNA_x(deg)",
    "FNA_y(deg)",
    "FNA_z(deg)",
]


def get_metric_formatted_row(nifti_file, metrics_dict):
    """output formatted string containing comma-separated metrics"""
    return f'{get_nifti_stem(str(nifti_file))},{metrics_dict["fhr"]:.2f},{metrics_dict["nsa"]:.2f},{metrics_dict["fo"]:.2f},{metrics_dict["fhc_x"]:.2f},{metrics_dict["fhc_y"]:.2f},{metrics_dict["fhc_z"]:.2f},{metrics_dict["fda_x"]:.2f},{metrics_dict["fda_y"]:.2f},{metrics_dict["fda_z"]:.2f},{metrics_dict["fna_x"]:.2f},{metrics_dict["fna_y"]:.2f},{metrics_dict["fna_z"]:.2f}\n'


def femur_morphometry_comparison_helper(
    filepath_1, filepath_2, log_dir=FEMUR_MANUAL_CUT_PLANE_DIR
):
    """helper func"""
    if isinstance(filepath_1, Path):
        filepath_1 = str(filepath_1)
    if isinstance(filepath_2, Path):
        filepath_2 = str(filepath_2)
    if not seg_contain_subtrochanter(filepath_1):
        return
    if not seg_contain_subtrochanter(filepath_2):
        return

    metrics_dict_v1 = get_femur_morphometry(
        filepath_1, get_subtrochanter_center(filepath_1), robust=True
    )
    metrics_dict_v2 = get_femur_morphometry(
        filepath_2, get_subtrochanter_center(filepath_2), robust=True
    )
    with open(f"{log_dir}/metrics_log/femur_clinical_variability.csv", "a+") as f:
        f.write(get_metric_formatted_row(filepath_1, metrics_dict_v1))
        f.write(get_metric_formatted_row(filepath_2, metrics_dict_v2))


def multiprocessing_helper(file):
    """multiprocessing helper"""
    file_prefix = get_nifti_stem(file)[:-2]

    v2_f = list(Path(FEMUR_MANUAL_CUT_PLANE_DIR).glob(f"{file_prefix}.nii.gz"))
    assert len(v2_f) == 1, f"expected 1 file got {len(v2_f)}"
    print(get_nifti_stem(file), get_nifti_stem(v2_f[0]))
    femur_morphometry_comparison_helper(file, v2_f[0])


def write_metric_log_header(filepath, filename="femur_clinical.csv"):
    """write output log header"""
    outdir = Path(f"{filepath}/metrics_log")
    outdir.mkdir(exist_ok=True)
    with open(outdir / filename, "w") as filestream:
        filestream_writer = csv.writer(filestream)

        filestream_writer.writerow(header)


if __name__ == "__main__":
    from multiprocessing import Pool
    import os

    write_metric_log_header(
        FEMUR_MANUAL_CUT_PLANE_DIR, "femur_clinical_variability.csv"
    )
    gt_files = sorted(list(Path(FEMUR_MANUAL_CUT_PLANE_DIR).glob("*_gt_2.nii.gz")))
    print(f"found {len(gt_files)} files")

    worker_fn = multiprocessing_helper
    num_workers = os.cpu_count() // 2
    pool = Pool(processes=num_workers)
    jobs = []
    for item in gt_files:
        job = pool.apply_async(worker_fn, (item,))
        jobs.append(job)
    for job in jobs:
        job.get()
    pool.close()
    pool.join()
