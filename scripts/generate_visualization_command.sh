for anatomy in femur vertebra rib hip
do
    echo $anatomy
    python scripts/comparative_visualize_v2.py --anatomy $anatomy --base_model SwinUNETR > scripts/comparative_visualize_$anatomy.sh
    sh scripts/comparative_visualize_$anatomy.sh
    python scripts/tile_images.py --anatomy "$anatomy"
done