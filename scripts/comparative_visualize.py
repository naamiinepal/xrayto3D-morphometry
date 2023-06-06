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
    quantile_25th_subject = df[df.DSC == df.quantile(q=0.25, numeric_only=True)["DSC"]][
        "subject-id"
    ]
    quantile_75th_subject = df[df.DSC == df.quantile(q=0.75, numeric_only=True)["DSC"]][
        "subject-id"
    ]
    return (
        best_subject,
        quantile_75th_subject,
        median_subject,
        quantile_25th_subject,
        worst_subject,
    )


def get_visualization_command(
    ANATOMY, subject_id, run, orientation, model_name, subject_out_prefix
):
    in_file = f"2d-3d-benchmark/{run.id}/evaluation/{subject_id}_pred.nii.gz"

    out_file = f"results/{ANATOMY}/{model_name}/{orientation}/{subject_out_prefix}_{orientation}.png"
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    command_string = f"python scripts/preview.py {in_file}  --size 500 500 --color 255 193 149 --orientation {orientation} --projection orthographic --out {out_file}"
    return command_string


if __name__ == "__main__":
    (
        best_subject,
        quantile_75th_subject,
        median_subject,
        quantile_25th_subject,
        worst_subject,
    ) = get_best_median_worst_sample("SwinUNETR", runs)

    run = get_run_from_model_name("SwinUNETR", runs)
    for orientation in orientations:
        for subject_id, subject_id_prefix in zip(
            (
                best_subject,
                quantile_75th_subject,
                median_subject,
                quantile_25th_subject,
                worst_subject,
            ),
            ("best", "quantile_75", "median", "quantile_25", "worst"),
        ):
            gt_command = get_visualization_command(
                ANATOMY, subject_id, run, orientation, "groundtruth", subject_id_prefix
            )
            print(gt_command)

    for orientation in orientations:
        for model_name in MODEL_NAMES:
            run = get_run_from_model_name(model_name, runs)
            for subject_id, subject_id_prefix in zip(
                (
                    best_subject,
                    quantile_75th_subject,
                    median_subject,
                    quantile_25th_subject,
                    worst_subject,
                ),
                ("best", "quantile_75", "median", "quantile_25", "worst"),
            ):
                gt_command = get_visualization_command(
                    ANATOMY, subject_id, run, orientation, model_name, subject_id_prefix
                )
                print(gt_command)
