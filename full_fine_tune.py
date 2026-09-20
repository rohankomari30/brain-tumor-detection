"""
Brain Tumor Detection - Full Fine-tuning Script
Unlike fine_tune.py (which only unfroze the top 30% of the base model),
this unfreezes the ENTIRE MobileNetV2 base and continues training at a very
low learning rate. This matches the methodology used in published papers
that report 96-99% accuracy on this dataset.

Run this AFTER train.py has already produced brain_tumor_model.h5.
This saves its result separately as transfer_model_full_finetuned.h5,
so it won't overwrite your existing model.

Usage:
    python full_fine_tune.py
    python full_fine_tune.py --epochs 15
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report


def parse_args():
    parser = argparse.ArgumentParser(description="Fully fine-tune the brain tumor classifier")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--model_path", default="brain_tumor_model.h5",
                         help="The already-trained transfer learning model to continue from")
    parser.add_argument("--save_path", default="transfer_model_full_finetuned.h5")
    return parser.parse_args()


def main():
    args = parse_args()
    train_dir = os.path.join(args.data_dir, "Training")
    test_dir = os.path.join(args.data_dir, "Testing")
    img_size = (224, 224)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading existing model from {args.model_path} ...")
    model = load_model(args.model_path)

    print("Loading data...")
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=args.batch_size,
        class_mode="categorical", subset="training", shuffle=True,
    )
    val_generator = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=args.batch_size,
        class_mode="categorical", subset="validation", shuffle=False,
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=args.batch_size,
        class_mode="categorical", shuffle=False,
    )
    class_names = list(train_generator.class_indices.keys())
    print("Classes found:", class_names)

    # ---- Unfreeze EVERY layer ----
    for layer in model.layers:
        layer.trainable = True
    print(f"Unfroze all {len(model.layers)} layers for full fine-tuning")

    # Very low learning rate is essential here - training every layer of a
    # pretrained network at a normal learning rate would destroy the useful
    # weights it already has. This is the standard approach papers use.
    model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        os.path.join(args.output_dir, "best_full_finetuned.h5"), monitor="val_accuracy", save_best_only=True
    )
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7)

    print(f"\nFull fine-tuning for up to {args.epochs} epochs (this will be slower per epoch than before)...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=[early_stop, checkpoint, reduce_lr],
    )

    # ---- Accuracy / loss curves ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Full Fine-tuning: Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Full Fine-tuning: Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    curves_path = os.path.join(args.output_dir, "full_finetune_accuracy_loss_curves.png")
    plt.savefig(curves_path, dpi=150)
    print(f"Saved {curves_path}")

    # ---- Test set evaluation ----
    print("\nEvaluating fully fine-tuned model on test set...")
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    test_generator.reset()
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Full Fine-tuning")
    cm_path = os.path.join(args.output_dir, "full_finetune_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Saved {cm_path}")

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report (fully fine-tuned):\n")
    print(report)

    report_path = os.path.join(args.output_dir, "full_finetune_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write(report)
    print(f"Saved {report_path}")

    model.save(args.save_path)
    print(f"\nSaved as {args.save_path}")
    print("\nCompare this against:")
    print("  - Baseline CNN: 87.25%")
    print("  - Partially fine-tuned transfer model: 81.12%")


if __name__ == "__main__":
    main()
