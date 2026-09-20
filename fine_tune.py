"""
Brain Tumor Detection - Fine-tuning Script
Continues training the model saved by train.py, unfreezing some of
MobileNetV2's top layers for a further accuracy boost. Run this AFTER
train.py has already produced brain_tumor_model.h5 - it picks up from there
instead of starting over.

Usage:
    python fine_tune.py
    python fine_tune.py --epochs 10
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the brain tumor classifier")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--model_path", default="brain_tumor_model.h5")
    parser.add_argument("--unfreeze_fraction", type=float, default=0.3,
                         help="Fraction of base model layers to unfreeze, counting from the end")
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

    # ---- Unfreeze the top portion of the base model ----
    # The last 4 layers are the custom classification head (GAP, Dense,
    # Dropout, Dense) added in train.py - everything before that is
    # MobileNetV2's original layers.
    head_layers = 4
    base_layer_count = len(model.layers) - head_layers
    fine_tune_at = int(base_layer_count * (1 - args.unfreeze_fraction))

    for i, layer in enumerate(model.layers):
        layer.trainable = i >= fine_tune_at

    print(f"Unfroze layers from index {fine_tune_at} onward (out of {len(model.layers)} total layers)")

    # Recompile with a much smaller learning rate - important when fine-tuning
    # pretrained weights, so we nudge them gently instead of overwriting what
    # they already learned from ImageNet.
    model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        os.path.join(args.output_dir, "best_model_finetuned.h5"), monitor="val_accuracy", save_best_only=True
    )

    print(f"\nFine-tuning for up to {args.epochs} epochs...")
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
    axes[0].set_title("Fine-tuning: Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Fine-tuning: Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    curves_path = os.path.join(args.output_dir, "finetune_accuracy_loss_curves.png")
    plt.savefig(curves_path, dpi=150)
    print(f"Saved {curves_path}")

    # ---- Test set evaluation ----
    print("\nEvaluating fine-tuned model on test set...")
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
    plt.title("Confusion Matrix (fine-tuned)")
    cm_path = os.path.join(args.output_dir, "finetune_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Saved {cm_path}")

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report (fine-tuned):\n")
    print(report)

    report_path = os.path.join(args.output_dir, "finetune_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write(report)
    print(f"Saved {report_path}")

    # ---- Overwrite the main model file so app.py picks up the improved version ----
    model.save(args.model_path)
    print(f"\nFine-tuned model saved as {args.model_path} — app.py will use this improved version.")


if __name__ == "__main__":
    main()
