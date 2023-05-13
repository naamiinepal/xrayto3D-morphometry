"""visualize femur morphometry algorithm"""
from pathlib import Path
import argparse
import os
from functools import partial
from multiprocessing import Pool

from xrayto3d_morphometry import (
    get_femur_morphometry,
    get_nifti_stem,
    get_subtrochanter_center,
    seg_contain_subtrochanter
)
FEMUR_MANUAL_CUT_PLANE_DIR = (
    "2D-3D-Reconstruction-Datasets/morphometry/femur_manual_cut_plane"
)


def handle_predicted_nifti(nifti_file):
    """predicted nifti do not have subtrochanter segmented"""
    prefix = '_pred'
    file_prefix = get_nifti_stem(nifti_file)[:-len(prefix)]
    manual_localization_file = list(Path(FEMUR_MANUAL_CUT_PLANE_DIR).glob(f'{file_prefix}_gt.nii.gz'))
    if len(manual_localization_file) != 1:
        print(
            f"Expected 1 but got {len(manual_localization_file)} files with prefix {file_prefix}"
        )
        return {}
    gt_filepath = str(manual_localization_file[0])
    if not seg_contain_subtrochanter(gt_filepath):
        return {}
    
    subtrochanter_center = get_subtrochanter_center(gt_filepath)
    metrics_dict = get_femur_morphometry(
        nifti_file, subtrochanter_center
    )
    return metrics_dict


def handle_groundtruth_nifti(nifti_file):
    """groundtruth nifti have subtrochanter segmented"""
    prefix = '_gt'
    file_prefix = get_nifti_stem(nifti_file)[:-len(prefix)]
    manual_localization_file = list(Path(FEMUR_MANUAL_CUT_PLANE_DIR).glob(f'{file_prefix}_gt.nii.gz'))
    if len(manual_localization_file) != 1:
        print(
            f"Expected 1 but got {len(manual_localization_file)} files with prefix {file_prefix}"
        )
        return {}
    gt_filepath = str(manual_localization_file[0])
    if not seg_contain_subtrochanter(gt_filepath):
        return {}
    subtrochanter_center = get_subtrochanter_center(gt_filepath)
    metrics_dict = get_femur_morphometry(gt_filepath, subtrochanter_center)
    return metrics_dict


def single_processing():
    parser = argparse.ArgumentParser()
    parser.add_argument('nifti_file')

    args = parser.parse_args()
    if 'pred' in args.nifti_file:
        # model prediction
        metrics_dict = handle_predicted_nifti(args.nifti_file)
    elif 'gt' in args.nifti_file:
        # groundtruth
        metrics_dict = handle_groundtruth_nifti(args.nifti_file)
    else: 
        raise ValueError('filetype should have either gt or pred, found none')
    print(metrics_dict)


def get_formatted_header():
    """return header for readability """
    header = ("id,gt_or_pred" +
              ",fhr,fhc_x,fhc_y,fhc_z" +
              ",nsa" +
              ",fna_x,fna_y,fna_z" +
              ",fda_x,fda_y,fda_z")
    return header


def get_formatted_row(nifti_file, measurements: dict):
    """return formatted string containing comma-separated measurements """
    return f"{get_nifti_stem(str(nifti_file))[:16]},{file_type_gt_or_pred(str(nifti_file))},{measurements['fhr']:.3f},{measurements['fhc_x']:.3f},{measurements['fhc_y']:.3f},{measurements['fhc_z']:.3f},{measurements['nsa']:.3f},{measurements['fna_x']:.3f},{measurements['fna_y']:.3f},{measurements['fna_z']:.3f},{measurements['fda_x']:.3f},{measurements['fda_y']:.3f},{measurements['fda_z']:.3f}"


def write_log_header(filepath, filename):
    """write output log header"""
    outdir = Path(filepath)
    outdir.mkdir(exist_ok=True)
    with open(outdir / filename, 'w', encoding='utf-8') as f:
        header = get_formatted_header()
        f.write(f'{header}\n')


def file_type_gt_or_pred(filename:str):
    """return either GT or PRED"""
    if 'gt' in filename:
        return 'GT'
    if 'pred' in filename:
        return 'PRED'
    raise ValueError(f'filename {filename} should either contain `gt` or `pred` as prefix')


def femur_morphometry_helper(nifti_filename, log_dir, log_filename):
    "helper func"
    nifti_filename = str(nifti_filename)
    if 'pred' in nifti_filename:
        # model prediction
        metrics_dict = handle_predicted_nifti(nifti_filename)
    elif 'gt' in nifti_filename:
        # groundtruth
        metrics_dict = handle_groundtruth_nifti(nifti_filename)
    else:
        raise ValueError(f'filename {nifti_filename} should either contain `gt` or `pred` as prefix')
    if not metrics_dict:
        print('Empty Dict')
    else:

        with open(f'{log_dir}/{log_filename}', 'a', encoding='utf-8') as f:
            row = get_formatted_row(nifti_filename, metrics_dict)
            f.write(f'{row}\n')


def process_dir_multithreaded():
    """process all files in a dir"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--log_filename', type=str)

    args = parser.parse_args()
    suffix = '*.nii.gz'
    filenames = sorted(list(Path(args.dir).glob(suffix)))
    print(f'processing {len(filenames)} files from {args.dir}')

    # write output file header
    write_log_header(args.dir, args.log_filename)
    worker_fn = partial(femur_morphometry_helper,
                        log_dir=args.dir,
                        log_filename=args.log_filename)
    num_workers = os.cpu_count()
    pool = Pool(processes=num_workers)
    jobs = []
    for item in filenames:
        job = pool.apply_async(worker_fn, (item,))
        jobs.append(job)
    for job in jobs:
        job.get()
    pool.close()
    pool.join()


if __name__ == '__main__':
    # single_processing()
    process_dir_multithreaded()
