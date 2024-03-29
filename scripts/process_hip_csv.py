"""
read a csv file containing hip morphometry measurements that contains both groundtruth and predicted 3d hip models identified by the *id* column. Subtract the corresponding columns and obtain the hip morphometry error that includes both reconstruction and measurement errors.

"""
import argparse
import pandas as pd
from pathlib import Path

from xrayto3d_morphometry import get_distance_between_points

parser = argparse.ArgumentParser()
parser.add_argument("csv_file")

args = parser.parse_args()

df = pd.read_csv(args.csv_file)
header = "id,ASIS_L,ASIS_R,PT_L,PT_R,IS_L,IS_R,PSIS_L,PSIS_R\n"


def get_formatted_row(id, landmarks):
    assert len(landmarks) == 8
    return f"{id},{landmarks[0]:.3f},{landmarks[1]:.3f},{landmarks[2]:.3f},{landmarks[3]:.3f},{landmarks[4]:.3f},{landmarks[5]:.3f},{landmarks[6]:.3f},{landmarks[7]:.3f}\n"


outfile = Path(args.csv_file).with_name("hip_landmark_error.csv")
with open(str(outfile), "w", encoding="utf-8") as f:
    f.write(header)
    ids = df["id"].unique()

    for id in ids:
        rows = df[df["id"] == id].to_numpy()
        if len(rows) != 2:
            print(f"subject: {id}exactly 2 rows expected, got {len(rows)}")
            continue
        gt_row = rows[0][2:]
        pred_row = rows[1][2:]
        diff = [
            get_distance_between_points(
                gt_row[i * 3 : i * 3 + 3], pred_row[i * 3 : i * 3 + 3]
            )
            for i in range(8)
        ]
        f.write(get_formatted_row(id, diff))
