import argparse
import pandas as pd

from xrayto3d_morphometry import get_distance_between_points

parser = argparse.ArgumentParser()
parser.add_argument('csv_file')

args = parser.parse_args()

df = pd.read_csv(args.csv_file, header=None)
header = 'id,ASIS_L,ASIS_R,PT_L,PT_R,IS_L,IS_R,PSIS_L,PSIS_R\n'


def get_formatted_row(id,landmarks):
    assert len(landmarks) == 7
    return f'{id},{landmarks[0]:.3f},{landmarks[1]:.3f},{landmarks[2]:.3f},{landmarks[3]:.3f},{landmarks[4]:.3f},{landmarks[5]:.3f},{landmarks[6]:.3f}\n'


with open('pelvic_landmark_error.csv', 'w', encoding='utf-8') as f:
    f.write(header)
    ids = df[0].unique()

    for id in ids:
        rows = df[df[0] == id].to_numpy()
        assert len(rows) == 2, f'exactly 2 rows expected'
        gt_row = rows[0][2:]
        pred_row = rows[1][2:]
        diff = [get_distance_between_points(gt_row[i*3:i*3+3],pred_row[i*3:i*3+3]) for i in range(7)]
        f.write(get_formatted_row(id, diff))
