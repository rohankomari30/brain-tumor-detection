"""
Brain Tumor Detection - Baseline CNN v2 (CLAHE preprocessing + stronger augmentation)
Third experiment: same baseline CNN architecture as train_baseline_cnn.py, but
with CLAHE contrast enhancement (a technique used in several published papers
on this dataset) and heavier data augmentation, plus batch normalization for
more stable training. Saved separately so it doesn't overwrite your other
models - lets you compare all of them in your report.

Requires opencv-python:
    pip install opencv-python

Usage:
    python train_baseline_cnn_v2.py
    python train_baseline_cnn_v2.py --epochs 25
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline CNN v2 with CLAHE + stronger augmentation")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="outputs")
    return parser.parse_args()


def apply_clahe(img):
    """Contrast-Limited Adaptive Histogram Equalization - enhances local
    contrast, which several published papers use as a preprocessing step
    for MRI classification. Applied to the grayscale version, then
    converted back to 3 channels since the model expects RGB input."""
    img = np.clip(img, 0, 255).astype("uint8")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return enhanced_rgb.astype("float32")


def build_baseline_cnn_v2(num_classes):
    inputs = Input(shape=(224, 224, 3))

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    args = parse_args()
    train_dir = os.path.join(args.data_dir, "Training")
    test_dir = os.path.join(args.data_dir, "Testing")
    os.makedirs(args.output_dir, exist_ok=True)

    img_size = (224, 224)

    print("Loading data with CLAHE preprocessing + stronger augmentation...")
    train_datagen = ImageDataGenerator(
        preprocessing_function=apply_clahe,
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        validation_split=0.2,
    )
    test_datagen = ImageDataGenerator(preprocessing_function=apply_clahe, rescale=1.0 / 255)

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

    print("\nBuilding baseline CNN v2 (CLAHE + augmentation + batch norm)...")
    model = build_baseline_cnn_v2(num_classes=len(class_names))
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        os.path.join(args.output_dir, "best_baseline_cnn_v2.h5"), monitor="val_accuracy", save_best_only=True
    )
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

    print(f"\nTraining for up to {args.epochs} epochs...")
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
    axes[0].set_title("Baseline CNN v2 (CLAHE): Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Baseline CNN v2 (CLAHE): Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    curves_path = os.path.join(args.output_dir, "baseline_v2_accuracy_loss_curves.png")
    plt.savefig(curves_path, dpi=150)
    print(f"Saved {curves_path}")

    # ---- Test set evaluation ----
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    test_generator.reset()
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Baseline CNN v2 (CLAHE)")
    cm_path = os.path.join(args.output_dir, "baseline_v2_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Saved {cm_path}")

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:\n")
    print(report)

    report_path = os.path.join(args.output_dir, "baseline_v2_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Baseline CNN v2 (CLAHE) Test Accuracy: {test_acc * 100:.2f}%\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write(report)
    print(f"Saved {report_path}")

    model.save("baseline_cnn_v2_model.h5")
    print("\nSaved as baseline_cnn_v2_model.h5")
    print("\nCompare against:")
    print("  - Baseline CNN (original): 87.25%")
    print("  - Partial fine-tune: 81.12%")
    print("  - Full fine-tune: 83.75%")


if __name__ == "__main__":
    main()
