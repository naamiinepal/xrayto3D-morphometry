"""Tile images"""
from PIL import Image, ImageFont, ImageDraw
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--anatomy")

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
    "TLPredictor": "TLEmbedding",
}
GRID_COL = len(MODEL_NAMES)  # MODELS
GRID_ROW = 2  # MODELS + GROUNDTRUTH
SINGLE_IMG_SZ = 500
ANATOMY = args.anatomy


def write_text_on_img(model_name, model_img, coords, align="left"):
    draw = ImageDraw.Draw(model_img)
    font = ImageFont.truetype(
        "/usr/share/texmf/fonts/opentype/public/lm/lmsans12-regular.otf", 70
    )
    draw.text(coords, model_name, (0, 0, 0), font=font, align=align)


def save_montage(ANATOMY, subject_type):
    for orientation in orientations:
        MONTAGE = Image.new("RGB", (SINGLE_IMG_SZ * GRID_COL, SINGLE_IMG_SZ * GRID_ROW))

        for i, model_name in enumerate(MODEL_NAMES):
            gt_file = f"results/{ANATOMY}/{model_name}/groundtruth/{orientation}/{subject_type}_{orientation}.png"
            pred_file = f"results/{ANATOMY}/{model_name}/predicted/{orientation}/{subject_type}_{orientation}.png"
            gt_img = Image.open(gt_file)
            pred_img = Image.open(pred_file)
            MONTAGE.paste(gt_img, (SINGLE_IMG_SZ * i, 0))
            MONTAGE.paste(pred_img, (SINGLE_IMG_SZ * i, SINGLE_IMG_SZ))
            if orientation == "axial" and subject_type == "quantile_75":
                if i == 0:
                    write_text_on_img(
                        "GROUNDTRUTH",
                        MONTAGE,
                        (0, 0),
                        align="center",
                    )
                write_text_on_img(
                    MODEL_NAMES_ALIAS[model_name],
                    MONTAGE,
                    (SINGLE_IMG_SZ * i, SINGLE_IMG_SZ),
                    align="center",
                )
        montage_out = f"results/{ANATOMY}/{subject_type}_{orientation}.png"
        MONTAGE.save(montage_out)


for subject_type in ("best", "quantile_75", "median", "quantile_25", "worst"):
    # for subject_type in ("quantile_75", "median", "quantile_25"):
    save_montage(ANATOMY, subject_type=subject_type)
