for file in 2d-3d-benchmark/p3qkfyj5/evaluation/*.nii.gz
do
    python vertebra_landmarks.py $file
done