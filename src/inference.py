"""
Usage: Choose a directory in data oem_sar to save --result_dir ong data in, including:
    - colored: colored output of the model
    - overlay: overlay of the colored output and the original SAR image
"""

import argparse
import pathlib
import cv2
from PIL import Image
from Semantic_Segmentation.source import dataset
from Semantic_Segmentation import test
import subprocess



DATA_ROOT = pathlib.Path(__file__).resolve().parent.parent / "data/oem_sar"

IMG_ROOT = DATA_ROOT / "test/sar_images"
LABEL_ROOT = DATA_ROOT / "test/labels"


def run_inference(result_root):
    
    subprocess.run(
        ["python", 
         "external/oem_sar/Semantic_Segmentation/test.py",
         "--data_root", str(IMG_ROOT),
         "--pretrained_model", "external/oem_sar/Semantic_Segmentation/pretrained/SAR_Mix_5_u-efficientnet-b4.pth",
         "--save_results", str(result_root)
        ]
                   )


def color_model_output(result_root):
    colored_dir = DATA_ROOT / result_root / "colored"
    overlay_dir = DATA_ROOT / result_root / "overlay"
    colored_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for png_path in result_root.glob("*.png"):
        label = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if label is None:
            continue
        if label.ndim == 3:
            label = label[..., 0]

        rgb = test.label2rgb(label)
        out_colored_path = colored_dir / f"colored_{png_path.name}"
        Image.fromarray(rgb).save(out_colored_path)
        print(f"Saved colored output to: {str(out_colored_path.name)}")

        sar_image_path = IMG_ROOT / f"{png_path.stem}.tif"
        if not sar_image_path.exists():
            print(f"Warning: No matching SAR image found for {png_path.name}, expected {sar_image_path.name}")
            continue

        sar_img = dataset.load_grayscale(sar_image_path)
        if sar_img is None:
            print(f"Warning: Could not load SAR image {sar_image_path}")
            continue

        sar_rgb = cv2.cvtColor(sar_img, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(sar_rgb, 0.7, rgb, 0.3, 0)

        out_overlay_path = overlay_dir / f"overlay_{png_path.name}"
        Image.fromarray(overlay).save(out_overlay_path)
        print(f"Saved overlay output to: {str(out_overlay_path.name)}")



def main(args):

    result_root = DATA_ROOT / args.results

    run_inference(result_root)
    color_model_output(result_root=result_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Inference and Coloring')
    parser.add_argument('--results', default="results", help='Prediction results directory name under data root')

    args = parser.parse_args()
    main(args)