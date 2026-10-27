"""
Brain Tumor Detection - Full Cross-Dataset Evaluation (all 5 experiments)
Evaluates ALL FOUR individually-trained models, plus the ensemble, on the
independent BDNeuro-MRI dataset - no retraining, just testing what's
already trained against genuinely unseen, independently-sourced data.

Requires all four saved model files to be present in this folder:
    baseline_cnn_model.h5           (Experiment: Baseline CNN, from scratch)
    brain_tumor_model.h5            (Experiment: MobileNetV2, frozen base)
    transfer_model_full_finetuned.h5 (Experiment: MobileNetV2, fully fine-tuned)
    baseline_cnn_v2_model.h5        (Experiment: CNN + CLAHE + augmentation)

Usage:
    python evaluate_cross_dataset_all.py --data_dir path/to/BDNeuro-MRI/test
"""

import os
import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def apply_clahe(img):
    """Same CLAHE preprocessing used to train baseline_cnn_v2_model.h5 -
    must be applied identically here or that model's results would be
    invalid (it was never trained on plain, non-CLAHE images)."""
    img = img.astype("uint8")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return enhanced_rgb.astype("float32")


def get_predictions(model, data_dir, use_clahe=False):
    if use_clahe:
        datagen = ImageDataGenerator(preprocessing_function=apply_clahe, rescale=1.0 / 255)
    else:
        datagen = ImageDataGenerator(rescale=1.0 / 255)

    generator = datagen.flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
    )
    preds = model.predict(generator, verbose=0)
    return preds, generator.classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to the external dataset's test folder")
    args = parser.parse_args()

    results = {}

    print("=== 1/4: Baseline CNN (from scratch) ===")
    baseline_model = load_model("baseline_cnn_model.h5")
    baseline_preds, y_true = get_predictions(baseline_model, args.data_dir)
    baseline_acc = accuracy_score(y_true, np.argmax(baseline_preds, axis=1))
    results["Baseline CNN (from scratch)"] = baseline_acc
    print(f"Accuracy: {baseline_acc*100:.2f}%\n")

    print("=== 2/4: MobileNetV2 (frozen base) ===")
    frozen_model = load_model("brain_tumor_model.h5")
    frozen_preds, _ = get_predictions(frozen_model, args.data_dir)
    frozen_acc = accuracy_score(y_true, np.argmax(frozen_preds, axis=1))
    results["MobileNetV2 (frozen base)"] = frozen_acc
    print(f"Accuracy: {frozen_acc*100:.2f}%\n")

    print("=== 3/4: MobileNetV2 (fully fine-tuned) ===")
    finetuned_model = load_model("transfer_model_full_finetuned.h5")
    finetuned_preds, _ = get_predictions(finetuned_model, args.data_dir)
    finetuned_acc = accuracy_score(y_true, np.argmax(finetuned_preds, axis=1))
    results["MobileNetV2 (fully fine-tuned)"] = finetuned_acc
    print(f"Accuracy: {finetuned_acc*100:.2f}%\n")

    print("=== 4/4: CNN + CLAHE + Augmentation ===")
    clahe_model = load_model("baseline_cnn_v2_model.h5")
    clahe_preds, y_true_clahe = get_predictions(clahe_model, args.data_dir, use_clahe=True)
    clahe_acc = accuracy_score(y_true_clahe, np.argmax(clahe_preds, axis=1))
    results["CNN + CLAHE + Augmentation"] = clahe_acc
    print(f"Accuracy: {clahe_acc*100:.2f}%\n")

    print("=== Ensemble (Baseline + Fine-tuned) ===")
    ensemble_preds = (baseline_preds + finetuned_preds) / 2.0
    ensemble_acc = accuracy_score(y_true, np.argmax(ensemble_preds, axis=1))
    results["Ensemble (Baseline + Fine-tuned)"] = ensemble_acc
    print(f"Accuracy: {ensemble_acc*100:.2f}%\n")

    print("\n=== SUMMARY: Cross-Dataset Accuracy (BDNeuro-MRI, all 5 experiments) ===")
    for name, acc in results.items():
        marker = " <== FINAL MODEL" if "Ensemble" in name else ""
        print(f"{name:<40}{acc*100:>6.2f}%{marker}")

    with open("outputs/cross_dataset_all_results.txt", "w") as f:
        f.write("Cross-Dataset Accuracy (BDNeuro-MRI test set)\n\n")
        for name, acc in results.items():
            f.write(f"{name}: {acc*100:.2f}%\n")
    print("\nSaved outputs/cross_dataset_all_results.txt")


if __name__ == "__main__":
    main()
