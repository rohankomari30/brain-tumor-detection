"""
Brain Tumor Detection - Full Evaluation (ROC-AUC, PR curves, ablation summary)
Extends evaluate_ensemble.py with metrics that go beyond accuracy - important
for a medical imaging classifier. Produces:
  - ROC curves (one-vs-rest, per class + macro-average)
  - Precision-Recall curves
  - Specificity/Sensitivity per class
  - A formatted ablation study table across all 5 experiments

Usage:
    python evaluate_full_metrics.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import label_binarize

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def main():
    test_dir = os.path.join(DATA_DIR, "Testing")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading models...")
    baseline_model = load_model("baseline_cnn_model.h5")
    transfer_model = load_model("transfer_model_full_finetuned.h5")

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
    )
    y_true = test_generator.classes
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])

    print("Running both models...")
    test_generator.reset()
    baseline_preds = baseline_model.predict(test_generator, verbose=1)
    test_generator.reset()
    transfer_preds = transfer_model.predict(test_generator, verbose=1)
    ensemble_preds = (baseline_preds + transfer_preds) / 2.0

    # ---- ROC curves (one-vs-rest per class + macro average) ----
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], ensemble_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(4):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 4
    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(8, 7))
    colors = ["#3FD0C9", "#FF6B6B", "#4A90D9", "#F5A623"]
    for i in range(4):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f"{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})")
    plt.plot(all_fpr, mean_tpr, color="black", linestyle="--", lw=2,
             label=f"Macro-average (AUC = {roc_auc['macro']:.3f})")
    plt.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Ensemble Model (One-vs-Rest)")
    plt.legend(loc="lower right", fontsize=9)
    roc_path = os.path.join(OUTPUT_DIR, "ensemble_roc_curves.png")
    plt.savefig(roc_path, dpi=150)
    print(f"Saved {roc_path}")

    # ---- Precision-Recall curves ----
    plt.figure(figsize=(8, 7))
    for i in range(4):
        prec, rec, _ = precision_recall_curve(y_true_bin[:, i], ensemble_preds[:, i])
        plt.plot(rec, prec, color=colors[i], lw=2, label=CLASS_NAMES[i])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - Ensemble Model (One-vs-Rest)")
    plt.legend(loc="lower left", fontsize=9)
    pr_path = os.path.join(OUTPUT_DIR, "ensemble_pr_curves.png")
    plt.savefig(pr_path, dpi=150)
    print(f"Saved {pr_path}")

    # ---- Sensitivity / Specificity per class (one-vs-rest) ----
    y_pred = np.argmax(ensemble_preds, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    print("\nPer-class Sensitivity (Recall) and Specificity:")
    lines = ["Class,Sensitivity,Specificity,ROC-AUC"]
    for i in range(4):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"  {CLASS_NAMES[i]:<12} Sensitivity: {sensitivity:.3f}   Specificity: {specificity:.3f}   ROC-AUC: {roc_auc[i]:.3f}")
        lines.append(f"{CLASS_NAMES[i]},{sensitivity:.3f},{specificity:.3f},{roc_auc[i]:.3f}")

    with open(os.path.join(OUTPUT_DIR, "sensitivity_specificity.csv"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved {os.path.join(OUTPUT_DIR, 'sensitivity_specificity.csv')}")

    print(f"\nMacro-average ROC-AUC: {roc_auc['macro']:.3f}")


if __name__ == "__main__":
    main()
