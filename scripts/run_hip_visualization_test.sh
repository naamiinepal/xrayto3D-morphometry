for file in 2d-3d-benchmark/sbclg22x/evaluation/*.nii.gz
do
    python hip_landmarks_v2.py $file --offscreen
done