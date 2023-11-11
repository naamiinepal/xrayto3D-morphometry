"""
read a csv file containing femur morphometry measurements that contains both groundtruth and predicted 3d femur models identified by the *id* column. Subtract the corresponding columns and obtain the femur morphometry error that includes both reconstruction and measurement errors.

"""
import argparse
import pandas as pd
from pathlib import Path

from xrayto3d_morphometry import get_distance_between_points

parser = argparse.ArgumentParser()
parser.add_argument('csv_file')

args = parser.parse_args()

df = pd.read_csv(args.csv_file)
header = 'id,FHR,FHC,NSA,FNA_x,FNA_y,FNA_z,FDA_x,FDA_y,FDA_z\n'


def get_formatted_row(id, measurements):
    assert len(measurements) == 9, f'Expected 9 but got {len(measurements)} columns'
    return f'{id},{measurements[0]:.3f},{measurements[1]:.3f},{measurements[2]:.3f},{measurements[3]:.3f},{measurements[4]:.3f},{measurements[5]:.3f},{measurements[6]:.3f},{measurements[7]:.3f},{measurements[8]:.3f}\n'


outfile = Path(args.csv_file).with_name('femur_morphometry_error.csv')
with open(str(outfile), 'w', encoding='utf-8') as f:
    f.write(header)
    ids = df['id'].unique()

    for id in ids:
        rows = df[df['id'] == id].to_numpy()
        assert len(rows) == 2, 'exactly 2 rows expected'
        gt_row = rows[0][2:]
        pred_row = rows[1][2:]
        diff = [abs(gt_row[0] - pred_row[0]),
                get_distance_between_points(gt_row[1:4], pred_row[1:4]),
                *[
                    abs(gt_row[i]-pred_row[i]) for i in range(4, 11)
                 ]
                ]
        f.write(get_formatted_row(id, diff))
