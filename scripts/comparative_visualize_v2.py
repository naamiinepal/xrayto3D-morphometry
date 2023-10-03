import wandb
from xrayto3d_morphometry import filter_wandb_run, get_run_from_model_name

import pandas as pd
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--anatomy")
args = parser.parse_args()

subdir = "evaluation"
ANATOMY = args.anatomy
tags = ["model-compare", "dropout"]
EVAL_LOG_CSV_PATH_TEMPLATE = "2d-3d-benchmark/{run_id}/{subdir}/metric-log.csv"


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
    quantile_25th_subject = get_quantile_row(df, quantile=0.25, column_name='DSC')[
        "subject-id"
    ].values[0]
    quantile_75th_subject = get_quantile_row(df, quantile=0.75, column_name='DSC')[
        "subject-id"
    ].values[0]
    return (
        best_subject,
        quantile_75th_subject,
        median_subject,
        quantile_25th_subject,
        worst_subject,
    )


def get_quantile_row(df, quantile: float, column_name: str):
    '''return row corresponding to the quantile value on the column
    the column should be numeric
    we have to choose the row with the closest quantile value'''
    return df.iloc[df.index.get_indexer([df.quantile(q=quantile, numeric_only=True)[column_name]],method='nearest')]


def get_visualization_command(
    ANATOMY,
    subject_id,
    run,
    orientation,
    model_name,
    subject_out_prefix,
    is_groundtruth=False,
):
    if is_groundtruth:
        in_file = f"2d-3d-benchmark/{run.id}/evaluation/{subject_id}_gt.nii.gz"
    else:
        in_file = f"2d-3d-benchmark/{run.id}/evaluation/{subject_id}_pred.nii.gz"
    gt_or_pred = "groundtruth" if is_groundtruth else "predicted"
    out_file = f"results/{ANATOMY}/{model_name}/{gt_or_pred}/{orientation}/{subject_out_prefix}_{orientation}.png"
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    command_string = f"python scripts/preview.py {in_file}  --size 500 500 --color 255 193 149 --orientation {orientation} --projection orthographic --out {out_file}"
    return command_string


if __name__ == "__main__":
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
    for orientation in orientations:
        for model_name in MODEL_NAMES:
            (
                best_subject,
                quantile_75th_subject,
                median_subject,
                quantile_25th_subject,
                worst_subject,
            ) = get_best_median_worst_sample(model_name, runs)
            run = get_run_from_model_name(model_name, runs)

            # save groundtruths and prediction
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
                    ANATOMY,
                    subject_id,
                    run,
                    orientation,
                    model_name,
                    subject_id_prefix,
                    is_groundtruth=True,
                )
                print(gt_command)
                pred_command = get_visualization_command(
                    ANATOMY,
                    subject_id,
                    run,
                    orientation,
                    model_name,
                    subject_id_prefix,
                    is_groundtruth=False,
                )
                print(pred_command)
