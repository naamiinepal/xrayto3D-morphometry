from tqdm import tqdm
from pathlib import Path
from xrayto3d_morphometry import (
    read_volume,
    get_segmentation_labels,
)


def seg_contain_subtrochanter(nifti_filename):
    seg_vol = read_volume(nifti_filename)
    label_indexes = get_segmentation_labels(seg_vol)
    label_dict = {"head": 4, "neck": 3, "sub_troc": 2}
    return label_dict["sub_troc"] in label_indexes


def process_dir():
    filenames = list(Path("femur_manual_cut_plane").glob("*_gt.nii.gz"))
    print(f"processing {len(filenames)} files")

    contains_neck_shaft_angle = 0
    for f in tqdm(filenames):
        if seg_contain_subtrochanter(f):
            contains_neck_shaft_angle += 1
    print(f"NSA {contains_neck_shaft_angle}")


if __name__ == "__main__":
    process_dir()
