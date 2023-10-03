# get comparative visualization of single subject
import wandb
from xrayto3d_morphometry import (
    filter_wandb_run,
    get_run_from_model_name,
)
from pathlib import Path
import argparse
import os

orientations = ["coronal", "sagittal", "axial"]
MODEL_NAMES = [
    "SwinUNETR",
    "UNETR",
    "AttentionUnet",
    "UNet",
    "MultiScale2DPermuteConcat",
    "TwoDPermuteConcat",
    "OneDConcat",
    "TLPredictor",
]


def get_visualization_command(
    ANATOMY, subject_id, run, orientation, model_name, subject_type
):
    out_file = (
        f"results/{ANATOMY}/{model_name}/{orientation}/{subject_type}_{orientation}.png"
    )
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    command_string = f"python scripts/preview.py 2d-3d-benchmark/{run.id}/evaluation/{subject_id}_pred.nii.gz --size 500 500 --color 255 193 149 --orientation {orientation} --projection orthographic --out {out_file}"
    return command_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anatomy")
    parser.add_argument("--base_model")
    parser.add_argument("--subject_id")
    args = parser.parse_args()
    subject_id = args.subject_id

    subdir = "evaluation"
    BASE_MODEL = args.base_model
    ANATOMY = args.anatomy
    tags = ["model-compare", "dropout"]
    EVAL_LOG_CSV_PATH_TEMPLATE = "2d-3d-benchmark/{run_id}/{subdir}/metric-log.csv"

    wandb.login()
    runs = filter_wandb_run(anatomy=ANATOMY, tags=tags)
    run = get_run_from_model_name("SwinUNETR", runs)
    for orientation in orientations:
        best_gt_in = f"2d-3d-benchmark/{run.id}/evaluation/{subject_id}_gt.nii.gz "
        best_gt_out = f"results/{ANATOMY}/groundtruth/{orientation}/{subject_id}_{orientation}.png"
        Path(best_gt_out).parent.mkdir(parents=True, exist_ok=True)
        gt_command = f"python scripts/preview.py {best_gt_in} --size 500 500 --color 255 193 149 --orientation {orientation} --projection orthographic --out {best_gt_out}"
        print(gt_command)
        os.system(gt_command)

    for orientation in orientations:
        for model_name in MODEL_NAMES:
            run = get_run_from_model_name(model_name, runs)
            vis_command = get_visualization_command(
                ANATOMY, subject_id, run, orientation, model_name, subject_id
            )

            print(vis_command)
            os.system(vis_command)
