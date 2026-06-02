"""
Usage: Choose a directory in data/oem_sar as --predictions to calculate and print metrics, being:
    - Pixel Accuracy
    - mIoU (Mean Intersection over Union)
    - Macro F1 Score
    - Macro Recall
    - Quantity Disagreement
    - Allocation Disagreement
"""

import argparse
import json
import pandas as pd
import numpy as np
from scipy import ndimage
from sklearn.metrics import confusion_matrix, classification_report, jaccard_score
import pathlib
import rasterio


def load_labels(path):
    src = rasterio.open(path, "r")
    return (src.read(1)).astype(np.uint8)


def pixel_accuracy(y_true, y_pred):
    return np.nanmean(y_true == y_pred)
    

def mIoU(y_pred, y_true, num_classes):

    iou_per_class = []

    for i in range(1, num_classes): # background is not included
        y_true_i = (y_true == i)
        y_pred_i = (y_pred == i)
        if np.sum(y_true_i) == 0:
            continue
        else:
            iou = jaccard_score(y_true_i, y_pred_i, average='binary', zero_division=0)
            iou_per_class.append(iou)

    return np.nanmean(iou_per_class)


def classification_report_metrics(y_true, y_pred, num_classes):
    report = classification_report(y_true, y_pred, labels=range(1, num_classes), output_dict=True, zero_division=0)

    macro_recall = report['macro avg']['recall']
    macro_precision = report['macro avg']['precision']
    mean_dice_macro_f1 = report['macro avg']['f1-score']

    return macro_precision, macro_recall, mean_dice_macro_f1


def boundary_mask(mask, width=1):
    """
    Return a binary boundary map for a single-class mask.
    width: how many pixels wide the boundary should be.
    """
    struct = np.ones((3, 3), dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=struct, border_value=0)
    boundary = mask & ~eroded
    if width > 1:
        dilate_struct = np.ones((2 * width + 1, 2 * width + 1), dtype=bool)
        boundary = ndimage.binary_dilation(boundary, structure=dilate_struct)
    return boundary

def boundary_iou(y_true, y_pred, num_classes, boundary_width=1):
    """
    Compute dataset-level boundary IoU.
    Accepts either:
    - y_true, y_pred: (H, W) single image
    - y_true, y_pred: (N, H, W) dataset
    """

    if y_true.ndim == 1 and y_pred.ndim == 1 and y_true.size == y_pred.size:
        # OpenEarthMap-SAR masks in this project are 1024x1024.
        side = 1024
        pixels_per_image = side * side
        total_pixels = y_true.size
        if total_pixels % pixels_per_image != 0:
            raise ValueError(
                "boundary_iou received flattened arrays with incompatible size for 1024x1024 masks."
            )
        n_images = total_pixels // pixels_per_image
        y_true = y_true.reshape(n_images, side, side)
        y_pred = y_pred.reshape(n_images, side, side)

    if y_true.ndim == 3:
        image_scores = []
        for true_mask, pred_mask in zip(y_true, y_pred):
            image_scores.append(
                boundary_iou(
                    true_mask,
                    pred_mask,
                    num_classes=num_classes,
                    boundary_width=boundary_width
                )
            )
        return np.nanmean(image_scores) if image_scores else 0.0

    ious = []

    for i in range(1, num_classes):
        true_boundary = boundary_mask(y_true == i, width=boundary_width)
        pred_boundary = boundary_mask(y_pred == i, width=boundary_width)

        union = np.logical_or(true_boundary, pred_boundary).sum()
        if union == 0:
            continue

        intersection = np.logical_and(true_boundary, pred_boundary).sum()
        ious.append(intersection / union)

    return np.nanmean(ious)


def disagreement_metrics(y_true, y_pred, num_classes):
    
    cm = confusion_matrix(y_true, y_pred, labels=range(1, num_classes))
    cm_norm = cm / np.sum(cm)  # Entry p_ij is proportion of total population

    quantity_disagreement = 0.0
    allocation_disagreement = 0.0
    
    for g in range(num_classes-1):  # Skip background class (0)
        # Marginal totals
        ref_total_g = np.sum(cm_norm[g, :])       # Row sum (Ground Truth total for class g)
        comp_total_g = np.sum(cm_norm[:, g])      # Col sum (Predicted total for class g)
        agreement_g = cm_norm[g, g]               # Diagonal (Correctly classified)
        
        # Quantity disagreement for class g
        q_g = abs(ref_total_g - comp_total_g)
        quantity_disagreement += q_g
        
        # Allocation disagreement for class g
        a_g = 2 * min(ref_total_g - agreement_g, comp_total_g - agreement_g)
        allocation_disagreement += a_g
        
    # Total disagreement parameters are halved because errors are shared between classes
    total_quantity_disagreement = quantity_disagreement / 2
    total_allocation_disagreement = allocation_disagreement / 2

    return total_quantity_disagreement, total_allocation_disagreement


def calculate_metrics(y_true, y_pred, num_classes):

    metrics = {}

    metrics.update({'Pixel Accuracy': pixel_accuracy(y_true, y_pred)})

    # Classification Report gives us Macro F1 (Mean Dice) and Macro Recall
    macro_precision, macro_recall, mean_dice_macro_f1 = classification_report_metrics(y_true, y_pred, num_classes=num_classes)
    metrics.update({
        'Macro Precision': macro_precision, 
        'Macro Recall': macro_recall, 
        'Mean Dice / Macro F1': mean_dice_macro_f1
        }
    )

    metrics.update({'mIoU': mIoU(y_pred, y_true, num_classes=num_classes)})
    metrics.update({'Boundary IoU': boundary_iou(y_true, y_pred, num_classes=num_classes, boundary_width=1)})

    # 2. Disagreement Metrics via Confusion Matrix Analysis
    total_quantity_disagreement, total_allocation_disagreement = disagreement_metrics(y_true, y_pred, num_classes=num_classes)
    metrics.update({
        'Quantity Disagreement': total_quantity_disagreement,
        'Allocation Disagreement': total_allocation_disagreement
    })

    return metrics


def load_metrics_file(path):
    path = pathlib.Path(path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_metrics_file(path, metrics):
    path = pathlib.Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

def normalize_metrics(metrics):
    return {k: float(v) for k, v in metrics.items()}

def evaluate_datasets(data_root, metrics_path, num_classes=9):
    data_root = pathlib.Path(data_root)
    metrics_path = pathlib.Path(metrics_path)

    existing_metrics = load_metrics_file(metrics_path)
    dataset_metrics = dict(existing_metrics)

    y_pred = np.array([load_labels(path) for path in sorted((data_root / "results").glob("*.png"), key=lambda p: p.name)]).flatten()

    # original dataset
    if "Original" not in dataset_metrics:
        y_true = np.array([load_labels(path) for path in sorted((data_root / "test/labels").glob("*.tif"), key=lambda p: p.name)]).flatten()
        metrics = calculate_metrics(y_true, y_pred, num_classes=num_classes)
        dataset_metrics["Original"] = normalize_metrics(metrics)
        save_metrics_file(metrics_path, dataset_metrics)
    else:
        print("Original metrics already present in", metrics_path)

    direct_subdirs = sorted([path for path in (data_root / "noise").iterdir() if path.is_dir()])
    for noise_dir in direct_subdirs:
        key = noise_dir.name
        if key in dataset_metrics:
            print(f"Skipping already-computed dataset: {key}")
            continue

        y_true = np.array([load_labels(path) for path in sorted(noise_dir.glob("*.png"), key=lambda p: p.name)]).flatten()

        metrics = calculate_metrics(y_true, y_pred, num_classes=num_classes)
        dataset_metrics[key] = normalize_metrics(metrics)
        save_metrics_file(metrics_path, dataset_metrics)

    return dataset_metrics



def main(args):
    DATA_ROOT = pathlib.Path.cwd().resolve() / "data/oem_sar"
    metrics_file = pathlib.Path(args.output_json)
    eval_dict = evaluate_datasets(DATA_ROOT, metrics_file, num_classes=args.num_classes)
    clean_dict = {pathlib.Path(k).name: v for k, v in eval_dict.items()}
    df = pd.DataFrame.from_dict(clean_dict, orient='index')
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Metrics Calculation')
    parser.add_argument('--predictions', default="results", help='Predictions directory name under data root')
    parser.add_argument('--num_classes', type=int, default=9, help='Number of classes')
    parser.add_argument('--output-json', default="metrics.json", help='JSON file to read/update metrics incrementally')

    args = parser.parse_args()
    main(args)
