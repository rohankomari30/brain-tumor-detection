"""
Brain Tumor Detection - Training Script
Run this locally (in VS Code, a terminal, etc.) to train the model.
It saves brain_tumor_model.h5 in this same folder, ready for app.py to load.

Expected dataset layout (see README.md for how to get this):
    data/
    ├── Training/
    │   ├── glioma_tumor/
    │   ├── meningioma_tumor/
    │   ├── no_tumor/
    │   └── pituitary_tumor/
    └── Testing/
        ├── glioma_tumor/
        ├── meningioma_tumor/
        ├── no_tumor/
        └── pituitary_tumor/

Usage:
    python train.py
    python train.py --epochs 20 --data_dir data
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # so it works without a display, saves plots straight to file
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report


def parse_args():
    parser = argparse.ArgumentParser(description="Train the brain tumor classifier")
    parser.add_argument("--data_dir", default="data", help="Folder containing Training/ and Testing/")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="outputs", help="Where to save plots/reports")
    return parser.parse_args()


def build_model(num_classes):
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False  # freeze pretrained layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    args = parse_args()
    train_dir = os.path.join(args.data_dir, "Training")
    test_dir = os.path.join(args.data_dir, "Testing")

    if not os.path.isdir(train_dir):
        raise SystemExit(
            f"\nCouldn't find {train_dir}\n"
            "Download the dataset first — see the 'Getting the dataset' section in README.md"
        )

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
    print("IMPORTANT: make sure this order matches CLASS_NAMES in app.py")

    print("\nBuilding model...")
    model = build_model(num_classes=len(class_names))
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        os.path.join(args.output_dir, "best_model.h5"), monitor="val_accuracy", save_best_only=True
    )

    print(f"\nTraining for up to {args.epochs} epochs...")
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
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    curves_path = os.path.join(args.output_dir, "accuracy_loss_curves.png")
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Saved {cm_path}")

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:\n")
    print(report)

    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write(report)
    print(f"Saved {report_path}")

    # ---- Save the model right next to app.py ----
    model_path = "brain_tumor_model.h5"
    model.save(model_path)
    print(f"\nModel saved as {model_path} — ready for app.py to load.")


if __name__ == "__main__":
    main()
