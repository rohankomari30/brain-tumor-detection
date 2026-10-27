"""
Brain Tumor Detection - Cross-Dataset Evaluation
Evaluates the already-trained ensemble on a SECOND, independent dataset
(not used in training at all) to test genuine generalization - this is
different from the out-of-distribution web-image tests, which were
individual/qualitative; this gives a real quantitative accuracy number
on a full external test set.

Dataset: "BDNeuro-MRI: A Bangladeshi Clinical Brain Tumor MRI Dataset for
Four-Class Deep Learning Classification" (Mendeley Data, DOI: 10.17632/zwr4ntf94j)
Download it from https://data.mendeley.com/datasets/zwr4ntf94j, unzip it,
and point --data_dir at its "test" split folder (containing Glioma/,
Meningioma/, Pituitary/, No_Tumor/ subfolders). Sourced from Epic & CSCR
Hospital, Bangladesh - a genuinely independent clinical source from your
training data, with a verified zero-overlap train/val/test split.

Usage:
    python evaluate_cross_dataset.py --data_dir path/to/mendeley_dataset/test
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to the external dataset's test folder")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading models...")
    baseline_model = load_model("baseline_cnn_model.h5")
    transfer_model = load_model("transfer_model_full_finetuned.h5")

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        args.data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
    )
    class_names = list(test_generator.class_indices.keys())
    print("Classes found in external dataset:", class_names)
    print("IMPORTANT: confirm this order matches ['glioma', 'meningioma', 'notumor', 'pituitary']")
    y_true = test_generator.classes

    print("\nRunning baseline CNN...")
    test_generator.reset()
    baseline_preds = baseline_model.predict(test_generator, verbose=1)

    print("\nRunning fine-tuned transfer model...")
    test_generator.reset()
    transfer_preds = transfer_model.predict(test_generator, verbose=1)

    ensemble_preds = (baseline_preds + transfer_preds) / 2.0
    y_pred = np.argmax(ensemble_preds, axis=1)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== Cross-Dataset (External) Test Accuracy: {acc*100:.2f}% ===\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - Cross-Dataset Evaluation ({acc*100:.2f}%)")
    cm_path = os.path.join(args.output_dir, "cross_dataset_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Saved {cm_path}")

    report_path = os.path.join(args.output_dir, "cross_dataset_report.txt")
    with open(report_path, "w") as f:
        f.write(f"External Dataset Test Accuracy: {acc * 100:.2f}%\n\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names))
    print(f"Saved {report_path}")

    print(f"\nCompare: same-dataset test accuracy was 88.06%. This run: {acc*100:.2f}%.")


if __name__ == "__main__":
    main()
