"""
Brain Tumor Detection - Baseline CNN (trained from scratch)
This is the "baseline" model for comparison against the transfer-learning
model in train.py. It uses a simple CNN with no pretrained weights, so you
can show a fair before/after comparison in your report.

Usage:
    python train_baseline_cnn.py
    python train_baseline_cnn.py --epochs 20
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report


def parse_args():
    parser = argparse.ArgumentParser(description="Train the baseline CNN (no transfer learning)")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="outputs")
    return parser.parse_args()


def build_baseline_cnn(num_classes):
    inputs = Input(shape=(224, 224, 3))

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
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

    print("\nBuilding baseline CNN (no pretrained weights)...")
    model = build_baseline_cnn(num_classes=len(class_names))
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        os.path.join(args.output_dir, "best_baseline_cnn.h5"), monitor="val_accuracy", save_best_only=True
    )

    print(f"\nTraining baseline for up to {args.epochs} epochs...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=[early_stop, checkpoint],
    )

    # ---- Accuracy / loss curves ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Baseline CNN: Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Baseline CNN: Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    curves_path = os.path.join(args.output_dir, "baseline_accuracy_loss_curves.png")
    plt.savefig(curves_path, dpi=150)
    print(f"Saved {curves_path}")

    # ---- Test set evaluation ----
    print("\nEvaluating baseline on test set...")
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Baseline Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Baseline Test Loss: {test_loss:.4f}")

    test_generator.reset()
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Baseline CNN")
    cm_path = os.path.join(args.output_dir, "baseline_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Saved {cm_path}")

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nBaseline Classification Report:\n")
    print(report)

    report_path = os.path.join(args.output_dir, "baseline_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Baseline CNN Test Accuracy: {test_acc * 100:.2f}%\n")
        f.write(f"Baseline CNN Test Loss: {test_loss:.4f}\n\n")
        f.write(report)
    print(f"Saved {report_path}")

    model.save("baseline_cnn_model.h5")
    print("\nBaseline model saved as baseline_cnn_model.h5")
    print("\nCompare this Test Accuracy against your transfer-learning model's")
    print("81.12% (in outputs/classification_report.txt) for your baseline-vs-proposed table.")


if __name__ == "__main__":
    main()
