# get worse, median and best cases
import wandb
from xrayto3d_morphometry import (
    filter_wandb_run,
    get_run_from_model_name,
)
import pandas as pd
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--anatomy")
parser.add_argument("--base_model")
args = parser.parse_args()

subdir = "evaluation"
BASE_MODEL = args.base_model
ANATOMY = args.anatomy
tags = ["model-compare", "dropout"]
EVAL_LOG_CSV_PATH_TEMPLATE = "2d-3d-benchmark/{run_id}/{subdir}/metric-log.csv"

wandb.login()
runs = filter_wandb_run(anatomy=ANATOMY, tags=tags)


orientations = ["coronal", "sagittal"]
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


def get_best_median_worst_sample(model_name, runs):
    run = get_run_from_model_name(model_name, runs)
    # read metric log csv
    csv_filename = EVAL_LOG_CSV_PATH_TEMPLATE.format(run_id=run.id, subdir=subdir)
    df = pd.read_csv(csv_filename)
    best_subject = df.nlargest(1, ["DSC"], "first")["subject-id"].values[0]
    worst_subject = df.nsmallest(1, ["DSC"], "first")["subject-id"].values[0]
    median_subject = df[df.DSC == df.median(numeric_only=True)["DSC"]][
        "subject-id"
    ].values[0]

    return best_subject, median_subject, worst_subject


def get_visualization_command(
    ANATOMY, subject_id, run, orientation, model_name, subject_type
):
    out_file = (
        f"results/{ANATOMY}/{model_name}/{orientation}/{subject_type}_{orientation}.png"
    )
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    command_string = f"python scripts/preview.py 2d-3d-benchmark/{run.id}/evaluation/{subject_id}_pred.nii.gz --size 500 500 --color 255 193 149 --orientation {orientation} --projection orthographic --out {out_file}"
    return command_string


best_subject, median_subject, worst_subject = get_best_median_worst_sample(
    "SwinUNETR", runs
)
run = get_run_from_model_name("SwinUNETR", runs)
for orientation in orientations:
    best_gt_in = f"2d-3d-benchmark/{run.id}/evaluation/{best_subject}_gt.nii.gz "
    best_gt_out = f"results/{ANATOMY}/groundtruth/{orientation}/best_{orientation}.png"
    Path(best_gt_out).parent.mkdir(parents=True, exist_ok=True)
    gt_command = f"python scripts/preview.py {best_gt_in} --size 500 500 --color 255 193 149 --orientation {orientation} --projection orthographic --out {best_gt_out}"
    print(gt_command)

    worst_gt_in = f"2d-3d-benchmark/{run.id}/evaluation/{worst_subject}_gt.nii.gz "
    worst_gt_out = (
        f"results/{ANATOMY}/groundtruth/{orientation}/worst_{orientation}.png"
    )
    Path(worst_gt_out).parent.mkdir(parents=True, exist_ok=True)
    gt_command = f"python scripts/preview.py {worst_gt_in} --size 500 500 --color 255 193 149 --orientation {orientation} --projection orthographic --out {worst_gt_out}"
    print(gt_command)

    median_gt_in = f"2d-3d-benchmark/{run.id}/evaluation/{median_subject}_gt.nii.gz "
    median_gt_out = (
        f"results/{ANATOMY}/groundtruth/{orientation}/median_{orientation}.png"
    )
    Path(median_gt_out).parent.mkdir(parents=True, exist_ok=True)
    gt_command = f"python scripts/preview.py {median_gt_in} --size 500 500 --color 255 193 149 --orientation {orientation} --projection orthographic --out {median_gt_out}"
    print(gt_command)

for orientation in orientations:
    for model_name in MODEL_NAMES:
        run = get_run_from_model_name(model_name, runs)
        best_command = get_visualization_command(
            ANATOMY, best_subject, run, orientation, model_name, "best"
        )

        print(best_command)

        worst_command = get_visualization_command(
            ANATOMY, worst_subject, run, orientation, model_name, "worst"
        )
        print(worst_command)

        median_command = get_visualization_command(
            ANATOMY, median_subject, run, orientation, model_name, "median"
        )
        print(median_command)
