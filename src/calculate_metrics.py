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
import pathlib
import re
import rasterio
import sqlite3

from scipy import ndimage
from sklearn.metrics import confusion_matrix, classification_report
from experiment_log import DEFAULT_DB_PATH, METRIC_COLUMNS, init_db, log_metric_updates


PER_CLASS_METRICS = {
    "precision",
    "recall",
    "dice_f1",
    "iou",
    "boundary_iou",
}
CLASSIFICATION_METRICS = {
    "precision",
    "recall",
    "dice_f1",
}
DISAGREEMENT_METRICS = {
    "quantity_disagreement",
    "allocation_disagreement",
}


def load_labels(path):
    src = rasterio.open(path, "r")
    return (src.read(1)).astype(np.uint8)


def ignore_gt_class(y_true, y_pred, ignore_class=0):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    valid = y_true != ignore_class
    return y_true[valid], y_pred[valid]


def pixel_accuracy(y_true, y_pred):
    y_true, y_pred = ignore_gt_class(y_true, y_pred)
    if y_true.size == 0:
        return np.nan
    return np.nanmean(y_true == y_pred)
    

def mIoU(y_pred, y_true, num_classes):

    iou_per_class = []
    y_true, y_pred = ignore_gt_class(y_true, y_pred)

    for i in range(1, num_classes): # background is not included
        y_true_i = (y_true == i)
        y_pred_i = (y_pred == i)
        union = np.logical_or(y_true_i, y_pred_i).sum()
        if union == 0:
            iou_per_class.append(np.nan)
        else:
            intersection = np.logical_and(y_true_i, y_pred_i).sum()
            iou_per_class.append(intersection / union)

    return np.asarray(iou_per_class, dtype=float)


def classification_report_metrics(y_true, y_pred, num_classes):
    y_true, y_pred = ignore_gt_class(y_true, y_pred)
    if y_true.size == 0:
        empty_scores = np.full(num_classes - 1, np.nan)
        return empty_scores, empty_scores.copy(), empty_scores.copy()
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
    return boundary

def boundary_iou(y_true, y_pred, num_classes, boundary_width=1):
    """
    Compute per-class boundary IoU from dataset-pooled boundary counts.

    Boundary intersections and unions are calculated within each image so
    boundaries never cross image edges.  Counts are then summed over images
    before division, weighting the result by boundary-union size.

    Accepts either:
    - y_true, y_pred: (H, W) single image
    - y_true, y_pred: (N, H, W) dataset
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

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

    if y_true.ndim == 2:
        y_true = y_true[np.newaxis, ...]
        y_pred = y_pred[np.newaxis, ...]

    total_intersections = np.zeros(num_classes - 1, dtype=np.int64)
    total_unions = np.zeros(num_classes - 1, dtype=np.int64)

    for true_mask, pred_mask in zip(y_true, y_pred):
        pred_mask = np.where(true_mask == 0, 0, pred_mask)
        for class_index, class_id in enumerate(range(1, num_classes)):
            true_boundary = boundary_mask(
                true_mask == class_id, width=boundary_width
            )
            pred_boundary = boundary_mask(
                pred_mask == class_id, width=boundary_width
            )
            total_intersections[class_index] += np.logical_and(
                true_boundary, pred_boundary
            ).sum()
            total_unions[class_index] += np.logical_or(
                true_boundary, pred_boundary
            ).sum()

    return np.divide(
        total_intersections,
        total_unions,
        out=np.full(num_classes - 1, np.nan, dtype=float),
        where=total_unions != 0,
    )


def disagreement_metrics(y_true, y_pred, num_classes):
    
    y_true, y_pred = ignore_gt_class(y_true, y_pred)
    if y_true.size == 0:
        return np.nan, np.nan
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    total = np.sum(cm)
    if total == 0:
        return np.nan, np.nan
    cm_norm = cm / total  # Entry p_ij is proportion of total population

    quantity_disagreement = 0.0
    allocation_disagreement = 0.0
    
    for g in range(num_classes):
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


def metric_status_by_tstamp(db_path):
    init_db(db_path)
    db_path = pathlib.Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    selected_columns = ", ".join(("creation_tstamp", *METRIC_COLUMNS))

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT {selected_columns}
            FROM experiment_logs
            """
        ).fetchall()

    status = {}
    for row in rows:
        creation_tstamp = row[0]
        status[creation_tstamp] = {
            metric_name
            for metric_name, value in zip(METRIC_COLUMNS, row[1:])
            if value is not None
        }
    return status


def calculate_missing_metrics(y_true, y_pred, num_classes, missing_metrics):
    missing_metrics = set(missing_metrics)
    metrics = {}

    if "pixel_accuracy" in missing_metrics:
        metrics["pixel_accuracy"] = pixel_accuracy(y_true, y_pred)

    classification_missing = CLASSIFICATION_METRICS & missing_metrics
    if classification_missing:
        precision, recall, dice_f1 = classification_report_metrics(
            y_true,
            y_pred,
            num_classes=num_classes,
        )
        classification_values = {
            "precision": precision,
            "recall": recall,
            "dice_f1": dice_f1,
        }
        metrics.update(
            {
                metric_name: classification_values[metric_name]
                for metric_name in classification_missing
            }
        )

    if "iou" in missing_metrics:
        metrics["iou"] = mIoU(y_pred, y_true, num_classes=num_classes)

    if "boundary_iou" in missing_metrics:
        metrics["boundary_iou"] = boundary_iou(
            y_true,
            y_pred,
            num_classes=num_classes,
            boundary_width=1,
        )

    disagreement_missing = DISAGREEMENT_METRICS & missing_metrics
    if disagreement_missing:
        quantity_disagreement, allocation_disagreement = disagreement_metrics(
            y_true,
            y_pred,
            num_classes=num_classes,
        )
        disagreement_values = {
            "quantity_disagreement": quantity_disagreement,
            "allocation_disagreement": allocation_disagreement,
        }
        metrics.update(
            {
                metric_name: disagreement_values[metric_name]
                for metric_name in disagreement_missing
            }
        )

    return metrics


def creation_tstamp_from_noise_dir(noise_dir):
    match = re.search(r"_(\d{6}_\d{6})$", pathlib.Path(noise_dir).name)
    if match is None:
        return None
    return match.group(1)


def evaluate_datasets(data_root, db_path=None, predictions="results", num_classes=9, pred_file_ending="*.png"):
    data_root = pathlib.Path(data_root)
    db_path = pathlib.Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    dataset_metrics = {}
    metric_status = metric_status_by_tstamp(db_path)
    all_metrics = set(METRIC_COLUMNS)

    prediction_dir = data_root / predictions
    # y_pred = np.array([load_labels(path) for path in sorted(prediction_dir.glob("*.png"), key=lambda p: p.name)]).flatten()
    y_pred = np.array([load_labels(path) for path in sorted(prediction_dir.glob(pred_file_ending), key=lambda p: p.name)]).flatten()


    noise_root = data_root / "noise"
    direct_subdirs = sorted([path for path in noise_root.iterdir() if path.is_dir()])
    for noise_dir in direct_subdirs:
        noisy_set_dirs = sorted([path for path in noise_dir.iterdir() if path.is_dir()])
        for noisy_set in noisy_set_dirs:
            creation_tstamp = creation_tstamp_from_noise_dir(noisy_set)
            if creation_tstamp is None:
                print(f"Skipping noise directory without timestamp suffix: {noisy_set.name}")
                continue

            existing_metrics = metric_status.get(creation_tstamp, set())
            missing_metrics = all_metrics - existing_metrics
            if not missing_metrics:
                print(f"Skipping dataset with complete metrics: {noisy_set.name}")
                continue

            y_true = np.array([load_labels(path) for path in sorted(noisy_set.glob("*.png"), key=lambda p: p.name)]).flatten()
    
            metrics = calculate_missing_metrics(
                y_true,
                y_pred,
                num_classes=num_classes,
                missing_metrics=missing_metrics,
            )
            normalized_metrics = normalize_metrics_for_db(metrics)
            log_metric_updates(
                creation_tstamp,
                normalized_metrics,
                db_path=db_path,
            )
            metric_status[creation_tstamp] = existing_metrics | set(normalized_metrics)
            dataset_metrics[noisy_set.name] = normalized_metrics

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
