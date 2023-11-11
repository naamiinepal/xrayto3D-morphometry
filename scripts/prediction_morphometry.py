"""model prediction morphometry"""

from typing import Tuple
from pathlib import Path
from xrayto3d_morphometry import (
    get_nifti_stem,
    femur_label_dict,
    get_segmentation_volume,
    extract_volume_surface,
)

FEMUR_MANUAL_CUT_PLANE_DIR = (
    "2D-3D-Reconstruction-Datasets/morphometry/femur_manual_cut_plane"
)


def get_groundtruth_file_from_prediction_file(predicted_nifti_file) -> Path:
    """return corresponding groundtruth segmentation file corresponding to model prediction"""
    suffix = "_pred"
    file_prefix = get_nifti_stem(predicted_nifti_file)[: -len(suffix)]
    groundtruth_file = list(
        Path(FEMUR_MANUAL_CUT_PLANE_DIR).glob(f"{file_prefix}*_gt.nii.gz")
    )
    assert (
        len(groundtruth_file) == 1
    ), f"Expected 1 but got {len(groundtruth_file)} files with prefix {file_prefix}"

    return groundtruth_file[0]


def get_prediction_file_from_groundtruth_file(
    groundtruth_nifti_file, run_id: str
) -> Path:
    """return corresponding predicted segmentation file corresponding to groundtruth segmentation"""
    suffix = "_gt"
    file_prefix = get_nifti_stem(groundtruth_nifti_file)[: -len(suffix)]
    predicted_file = list(
        Path(f"2d-3d-benchmark/{run_id}/evaluation").glob(f"{file_prefix}*_pred.nii.gz")
    )
    assert (
        len(predicted_file) == 1
    ), f"Expected 1 but fot {len(predicted_file)} files with prefix {file_prefix}"
    return predicted_file[0]


def get_subtrochanter_center(nifti_file) -> Tuple[float, float, float]:
    """get subtrochanter center-of-mass"""
    if isinstance(nifti_file, Path):
        nifti_file = str(nifti_file)

    # assume subtrochanter segmentation exists
    subtroc_mesh = extract_volume_surface(
        get_segmentation_volume(nifti_file, femur_label_dict["sub_troc"])
    )
    return subtroc_mesh.center_of_mass()
