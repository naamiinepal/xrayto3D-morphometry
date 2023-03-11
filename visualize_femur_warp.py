from xrayto3d_morphometry import *

def main():
    gt_fp = 'test_data/s0015_femur_left_msk_gt.nii.gz'
    pred_fp = 'test_data/s0015_femur_left_msk_pred.nii.gz'
    gt = get_mesh_from_segmentation(gt_fp)
    pred = get_mesh_from_segmentation(pred_fp)

    vedo.show(gt.c('blue',0.4),pred.c('red',0.4))
if __name__ == '__main__':
    main()