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
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, jaccard_score
import pathlib
import rasterio
import torch


def load_labels(path):
    src = rasterio.open(path, "r")
    return (src.read(1)).astype(np.uint8)


# metrics

def pixel_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def mIoU(y_pred, y_true, eps=1e-7, n_classes=9):

    pr = torch.from_numpy(y_pred).to(torch.uint8)
    gt = torch.from_numpy(y_true).to(torch.uint8)
     
    iou_per_class = []

    pr = pr.contiguous().view(-1)
    gt = gt.contiguous().view(-1)

    for sem_class in range(1, n_classes): # background is not included
        pr_inds = (pr == sem_class)
        gt_inds = (gt == sem_class)
        if gt_inds.long().sum().item() == 0:
            iou_per_class.append(np.nan)
        else: 
            intersect = torch.logical_and(pr_inds, gt_inds).sum().float().item()
            union = torch.logical_or(pr_inds, gt_inds).sum().float().item()
            iou = (intersect + eps) / (union + eps)
            iou_per_class.append(iou)
    return np.nanmean(iou_per_class)


def disagreement_metrics(y_true, y_pred, num_classes):
    
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    cm_norm = cm / np.sum(cm)  # Entry p_ij is proportion of total population
    
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
    
    # mIoU (Macro Jaccard Score)
    miou = mIoU(y_pred, y_true, n_classes=num_classes+1)


    y_true_flattened = y_true.flatten()
    y_pred_flattened = y_pred.flatten()

    p_accuracy = pixel_accuracy(y_true_flattened, y_pred_flattened)
    
    # Classification Report gives us Macro F1 (Mean Dice) and Macro Recall
    report = classification_report(y_true_flattened, y_pred_flattened, labels=range(num_classes), output_dict=True, zero_division=0)
    macro_recall = report['macro avg']['recall']
    mean_dice_macro_f1 = report['macro avg']['f1-score']
    
    # 2. Disagreement Metrics via Confusion Matrix Analysis
    total_quantity_disagreement, total_allocation_disagreement = disagreement_metrics(y_true_flattened, y_pred_flattened, num_classes)
    
    # Format and display output nicely
    print(f"--- Core Vision Metrics ---")
    print(f"Pixel Accuracy:                 {p_accuracy:.4f}")
    print(f"mIoU (Mean IoU):                {miou:.4f}")
    print(f"Macro Recall:                   {macro_recall:.4f}")
    print(f"Mean Dice / Macro F1:           {mean_dice_macro_f1:.4f}")
    
    print(f"\n--- Disagreement Metrics ---")
    print(f"Quantity Disagreement (Q):      {total_quantity_disagreement:.4f}")
    print(f"Allocation Disagreement (A):    {total_allocation_disagreement:.4f}")



def main(args):
    DATA_ROOT = pathlib.Path.cwd().resolve().parent / "data/oem_sar"
    results = args.predictions
    num_classes = 8

    y_true = np.array([load_labels(path) for path in (DATA_ROOT / "test/labels").glob("*.tif")])
    y_pred = np.array([load_labels(path) for path in (DATA_ROOT / results).glob("*.png")])

    print("\n")
    print(f"Metrics for {results.upper()}:")
    calculate_metrics(y_true, y_pred, num_classes=num_classes)
    print("\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Metrics Calculation')
    parser.add_argument('--predictions', default="results", help='Predictions directory name under data root')

    args = parser.parse_args()
    main(args)
