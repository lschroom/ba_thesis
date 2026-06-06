"""
Usage: Choose a directory in data/oem_sar as --predictions to calculate and print metrics, being:
    - Pixel Accuracy
    - IoU per foreground class
    - Precision per foreground class
    - Recall per foreground class
    - Dice / F1 per foreground class
    - Boundary IoU per foreground class
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
import re
import rasterio

from experiment_log import DEFAULT_DB_PATH, log_metrics


PER_CLASS_METRICS = {
    "precision",
    "recall",
    "dice_f1",
    "iou",
    "boundary_iou",
}


def load_labels(path):
    src = rasterio.open(path, "r")
    return (src.read(1)).astype(np.uint8)


def pixel_accuracy(y_true, y_pred):
    return np.nanmean(y_true == y_pred)
    

def mIoU(y_pred, y_true, num_classes):

    iou_per_class = []
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    for i in range(1, num_classes): # background is not included
        y_true_i = (y_true == i)
        y_pred_i = (y_pred == i)
        if np.sum(y_true_i) == 0:
            iou_per_class.append(np.nan)
        else:
            iou = jaccard_score(y_true_i, y_pred_i, average='binary', zero_division=0)
            iou_per_class.append(iou)

    return np.asarray(iou_per_class, dtype=float)


def classification_report_metrics(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    report = classification_report(y_true, y_pred, labels=range(1, num_classes), output_dict=True, zero_division=0)

    precision = []
    recall = []
    dice_f1 = []

    for i in range(1, num_classes):
        class_metrics = report[str(i)]
        precision.append(class_metrics['precision'])
        recall.append(class_metrics['recall'])
        dice_f1.append(class_metrics['f1-score'])

    return (
        np.asarray(precision, dtype=float),
        np.asarray(recall, dtype=float),
        np.asarray(dice_f1, dtype=float),
    )


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
        return np.nanmean(image_scores, axis=0) if image_scores else np.full(num_classes - 1, np.nan)

    ious = []

    for i in range(1, num_classes):
        true_boundary = boundary_mask(y_true == i, width=boundary_width)
        pred_boundary = boundary_mask(y_pred == i, width=boundary_width)

        union = np.logical_or(true_boundary, pred_boundary).sum()
        if union == 0:
            ious.append(np.nan)
            continue

        intersection = np.logical_and(true_boundary, pred_boundary).sum()
        ious.append(intersection / union)

    return np.asarray(ious, dtype=float)


def disagreement_metrics(y_true, y_pred, num_classes):
    
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
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

    metrics.update({'pixel_accuracy': pixel_accuracy(y_true, y_pred)})

    precision, recall, dice_f1 = classification_report_metrics(y_true, y_pred, num_classes=num_classes)
    metrics.update({
        'precision': precision,
        'recall': recall,
        'dice_f1': dice_f1
        }
    )

    metrics.update({'iou': mIoU(y_pred, y_true, num_classes=num_classes)})
    metrics.update({'boundary_iou': boundary_iou(y_true, y_pred, num_classes=num_classes, boundary_width=1)})

    # 2. Disagreement Metrics via Confusion Matrix Analysis
    total_quantity_disagreement, total_allocation_disagreement = disagreement_metrics(y_true, y_pred, num_classes=num_classes)
    metrics.update({
        'quantity_disagreement': total_quantity_disagreement,
        'allocation_disagreement': total_allocation_disagreement
    })

    return metrics


def _finite_float(value):
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _per_class_json(metric_name, values):
    values = np.asarray(values, dtype=float)
    return json.dumps(
        {
            f"{metric_name}_{class_id}": _finite_float(value)
            for class_id, value in enumerate(values, start=1)
        },
        sort_keys=True,
    )


def normalize_metrics_for_db(metrics):
    normalized = {}
    for key, value in metrics.items():
        if key in PER_CLASS_METRICS:
            normalized[key] = _per_class_json(key, value)
        else:
            normalized[key] = _finite_float(value)
    return normalized


def creation_tstamp_from_noise_dir(noise_dir):
    match = re.search(r"_(\d{6}_\d{6})$", pathlib.Path(noise_dir).name)
    if match is None:
        return None
    return match.group(1)


def evaluate_datasets(data_root, db_path=None, predictions="results", num_classes=9):
    data_root = pathlib.Path(data_root)
    db_path = pathlib.Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    dataset_metrics = {}

    prediction_dir = data_root / predictions
    y_pred = np.array([load_labels(path) for path in sorted(prediction_dir.glob("*.png"), key=lambda p: p.name)]).flatten()

    noise_root = data_root / "noise"
    direct_subdirs = sorted([path for path in noise_root.iterdir() if path.is_dir()])
    for noise_dir in direct_subdirs:
        creation_tstamp = creation_tstamp_from_noise_dir(noise_dir)
        if creation_tstamp is None:
            print(f"Skipping noise directory without timestamp suffix: {noise_dir.name}")
            continue

        y_true = np.array([load_labels(path) for path in sorted(noise_dir.glob("*.png"), key=lambda p: p.name)]).flatten()

        metrics = calculate_metrics(y_true, y_pred, num_classes=num_classes)
        normalized_metrics = normalize_metrics_for_db(metrics)
        log_metrics(
            creation_tstamp,
            normalized_metrics,
            db_path=db_path,
        )
        dataset_metrics[noise_dir.name] = normalized_metrics

    return dataset_metrics



def main(args):
    DATA_ROOT = pathlib.Path.cwd().resolve() / "data/oem_sar"
    eval_dict = evaluate_datasets(
        DATA_ROOT,
        db_path=args.db_path,
        predictions=args.predictions,
        num_classes=args.num_classes,
    )
    clean_dict = {pathlib.Path(k).name: v for k, v in eval_dict.items()}
    df = pd.DataFrame.from_dict(clean_dict, orient='index')
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Metrics Calculation')
    parser.add_argument('--predictions', default="results", help='Predictions directory name under data root')
    parser.add_argument('--num_classes', type=int, default=9, help='Number of classes')
    parser.add_argument('--db-path', default=DEFAULT_DB_PATH, help='SQLite DB file for experiment logs')

    args = parser.parse_args()
    main(args)
