for anatomy in femur vertebra rib hip
do
    echo $anatomy
    python scripts/comparative_visualize_v2.py --anatomy $anatomy > scripts/comparative_visualize_v2_$anatomy.sh
    sh scripts/comparative_visualize_v2_$anatomy.sh
    python scripts/tile_images_v2.py --anatomy "$anatomy"
done