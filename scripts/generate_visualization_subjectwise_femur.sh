gt_postfix=_gt.nii.gz
for file in 2d-3d-benchmark/0hoo4r74/evaluation/*$gt_postfix;
do
    base_name=$(basename ${file})
    subject_id=${base_name%_*} # remove everything after last _
    python scripts/comparative_visualize_subjectwise.py --anatomy femur --base_model SwinUNETR --subject_id $subject_id
    python scripts/tile_images_subjectwise.py --anatomy femur --subject_id $subject_id   
done