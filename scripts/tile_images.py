"""Tile images"""
from PIL import Image, ImageFont, ImageDraw
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--anatomy", type=str)

args = parser.parse_args()

Image.MAX_IMAGE_PIXELS = None

orientations = ["coronal", "sagittal", "axial"]
MODEL_NAMES = [
    "AttentionUnet",
    "SwinUNETR",
    "TwoDPermuteConcat",
    "MultiScale2DPermuteConcat",
    "UNet",
    "UNETR",
    "OneDConcat",
    "TLPredictor",
]
MODEL_NAMES_ALIAS = {
    "AttentionUnet": "AttUnet",
    "SwinUNETR": "SwinUNETR",
    "TwoDPermuteConcat": "2DConcat",
    "MultiScale2DPermuteConcat": "Multiscale2D",
    "UNet": "UNet",
    "UNETR": "UNETR",
    "OneDConcat": "1DConcat",
    "TLPredictor": "TLNet",
}
GRID_COL = len(MODEL_NAMES) + 1  # MODELS + GROUNDTRUTH
GRID_ROW = 1
SINGLE_IMG_SZ = 500
ANATOMY = args.anatomy


def save_montage(ANATOMY, subject_type):
    for orientation in orientations:
        MONTAGE = Image.new("RGB", (SINGLE_IMG_SZ * GRID_COL, SINGLE_IMG_SZ * GRID_ROW))

        gt_path = f"results/{ANATOMY}/groundtruth/{orientation}/{subject_type}_{orientation}.png"
        gt_img = Image.open(gt_path)
        MONTAGE.paste(gt_img, (0, 0))
        write_text_on_img("Groundtruth", MONTAGE, (0, 0))

        for i, model_name in enumerate(MODEL_NAMES):
            out_file = f"results/{ANATOMY}/{model_name}/{orientation}/{subject_type}_{orientation}.png"
            model_img = Image.open(out_file)
            MONTAGE.paste(model_img, (SINGLE_IMG_SZ * (i + 1), 0))
            write_text_on_img(
                MODEL_NAMES_ALIAS[model_name], MONTAGE, (SINGLE_IMG_SZ * (i + 1), 0)
            )

        montage_out = f"results/{ANATOMY}/{subject_type}_{orientation}.png"
        MONTAGE.save(montage_out)


def write_text_on_img(model_name, model_img, coords):
    draw = ImageDraw.Draw(model_img)
    font = ImageFont.truetype(
        "/usr/share/texmf/fonts/opentype/public/lm/lmmonocaps10-regular.otf", 70
    )
    draw.text(coords, model_name, (0, 0, 0), font=font)


for subject_type in ["median", "worst", "best"]:
    save_montage(ANATOMY, subject_type=subject_type)
