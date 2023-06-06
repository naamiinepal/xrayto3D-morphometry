import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("csv_file")

args = parser.parse_args()

df = pd.read_csv(args.csv_file)
header = "id,spl,spa,avbh,pvbh,svbl,ivbl,vcl\n"


def get_formatted_row(id, landmarks):
    assert len(landmarks) == 7
    return f"{id},{landmarks[0]:.3f},{landmarks[1]:.3f},{landmarks[2]:.3f},{landmarks[3]:.3f},{landmarks[4]:.3f},{landmarks[5]:.3f},{landmarks[6]:.3f}\n"


outfile = Path(args.csv_file).with_name("vertebra_morphometry_error.csv")
with open(str(outfile), "w", encoding="utf-8") as f:
    f.write(header)
    ids = df["id"].unique()

    for id in ids:
        rows = df[df["id"] == id].to_numpy()
        if len(rows) != 2:
            continue  # some cases might be failure cases and so one of groundtruth or predicted model result may not be available
        gt_row = rows[0][2:]
        pred_row = rows[1][2:]
        diff = [abs(gt_row[i] - pred_row[i]) for i in range(7)]
        f.write(get_formatted_row(id, diff))
