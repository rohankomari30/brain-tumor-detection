"""
Brain Tumor Detection - Ensemble Evaluation
Properly evaluates the ensemble (baseline CNN + fully fine-tuned transfer
model, averaged) on the held-out test set, producing real accuracy,
precision, recall, F1, and a confusion matrix - not placeholder numbers.

This does NOT retrain anything - it just loads your two already-trained
models and evaluates their combination properly, the way it should have
been evaluated from the start.

Usage:
    python evaluate_ensemble.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def main():
    test_dir = os.path.join(DATA_DIR, "Testing")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading models...")
    baseline_model = load_model("baseline_cnn_model.h5")
    transfer_model = load_model("transfer_model_full_finetuned.h5")

    # Clean, unaugmented test data - same as every other evaluation script
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
    )
    class_names = list(test_generator.class_indices.keys())
    y_true = test_generator.classes

    print("\nRunning baseline CNN on test set...")
    test_generator.reset()
    baseline_preds = baseline_model.predict(test_generator, verbose=1)

    print("\nRunning fine-tuned transfer model on test set...")
    test_generator.reset()
    transfer_preds = transfer_model.predict(test_generator, verbose=1)

    # ---- Individual model accuracies (sanity check against earlier runs) ----
    baseline_acc = accuracy_score(y_true, np.argmax(baseline_preds, axis=1))
    transfer_acc = accuracy_score(y_true, np.argmax(transfer_preds, axis=1))
    print(f"\nBaseline CNN test accuracy (sanity check): {baseline_acc * 100:.2f}%")
    print(f"Fine-tuned transfer model test accuracy (sanity check): {transfer_acc * 100:.2f}%")

    # ---- Ensemble: average the two probability distributions ----
    ensemble_preds = (baseline_preds + transfer_preds) / 2.0
    y_pred = np.argmax(ensemble_preds, axis=1)

    ensemble_acc = accuracy_score(y_true, y_pred)
    print(f"\nEnsemble test accuracy: {ensemble_acc * 100:.2f}%")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - Ensemble ({ensemble_acc*100:.2f}% accuracy)")
    cm_path = os.path.join(OUTPUT_DIR, "ensemble_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Saved {cm_path}")

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nEnsemble Classification Report:\n")
    print(report)

    report_path = os.path.join(OUTPUT_DIR, "ensemble_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Ensemble Test Accuracy: {ensemble_acc * 100:.2f}%\n\n")
        f.write("Individual model accuracies (for reference):\n")
        f.write(f"  Baseline CNN: {baseline_acc * 100:.2f}%\n")
        f.write(f"  Fine-tuned transfer model: {transfer_acc * 100:.2f}%\n\n")
        f.write(report)
    print(f"Saved {report_path}")

    print("\n=== Final comparison table ===")
    print(f"{'Model':<35}{'Test Accuracy':>15}")
    print(f"{'Baseline CNN':<35}{baseline_acc*100:>14.2f}%")
    print(f"{'Fine-tuned transfer model':<35}{transfer_acc*100:>14.2f}%")
    print(f"{'Ensemble (baseline + transfer)':<35}{ensemble_acc*100:>14.2f}%")


if __name__ == "__main__":
    main()
