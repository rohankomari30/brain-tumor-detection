"""
Brain Tumor Detection - Flask Backend
Loads TWO trained models (the baseline CNN and the fully fine-tuned transfer
learning model) and ensembles their predictions together, rather than
relying on a single model. Also flags predictions as low-confidence when
neither model is sure, and reports whether the two models agree - directly
addressing the out-of-distribution failure cases found during testing
(where a single model gave a confident-looking but wrong answer).

Also generates Grad-CAM heatmaps from BOTH models (baseline CNN and the
fully fine-tuned transfer model) showing which regions of the scan most
influenced each model's prediction, plus an independent occlusion
sensitivity map (baseline model) that cross-checks Grad-CAM using a
completely different, gradient-free technique. Serves a /performance page
summarizing all 5 experiments run for this project.
"""

import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import matplotlib.cm as cm
import tensorflow as tf

app = Flask(__name__)

# ---- Config ----
BASE_DIR = os.path.dirname(__file__)
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATHS = {
    "baseline": os.path.join(BASE_DIR, "baseline_cnn_model.h5"),
    "transfer": os.path.join(BASE_DIR, "transfer_model_full_finetuned.h5"),
}
IMG_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Below this confidence, we tell the user the result is unreliable instead
# of just presenting a guess as if it were certain.
LOW_CONFIDENCE_THRESHOLD = 70.0
HIGH_CONFIDENCE_THRESHOLD = 85.0

# A grayscale MRI slice has near-identical R, G, B values at every pixel.
# An ordinary color photo does not. This is a simple, explainable heuristic
# gate - not a trained OOD detector - and is documented as such; it will
# reject color photos but cannot catch every invalid input (e.g. a black
# and white photo of something else would pass this check).
COLOR_DIFF_REJECT_THRESHOLD = 18.0

# Class order MUST match the order Keras assigned during training.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

DISPLAY_NAMES = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

_models = {}  # loaded lazily, see load_models_once()
_last_conv_layer_cache = {}  # keyed by model name - see fix note below


def load_models_once():
    """Load both models into memory once, the first time either is needed."""
    global _models
    if not _models:
        from tensorflow.keras.models import load_model as keras_load_model

        for key, path in MODEL_PATHS.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model file not found at {path}. "
                    "Make sure both trained .h5 files are in this project's folder."
                )
            _models[key] = keras_load_model(path)
    return _models


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(file_bytes):
    """Resize + normalize an uploaded image to match training preprocessing."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, img


def check_image_validity(img_pil):
    """Lightweight heuristic gate: rejects images that clearly aren't
    grayscale MRI-style scans (e.g. an ordinary color photo). This is a
    simple statistical check, not a trained classifier - it catches the
    obvious cases (color photos, screenshots of unrelated content) but is
    not a robust, general-purpose OOD detector. Returns (is_valid, reason).
    """
    arr = np.array(img_pil.convert("RGB")).astype("float32")
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    color_diff = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b)))

    if color_diff > COLOR_DIFF_REJECT_THRESHOLD:
        return False, (
            "This image does not appear to be a grayscale MRI scan (it looks like "
            "a color photo). Please upload a brain MRI image."
        )

    # Flag near-blank / near-uniform images (corrupted uploads, solid colors)
    if float(np.std(arr)) < 5.0:
        return False, "This image appears blank or corrupted. Please upload a valid MRI scan."

    return True, None


def confidence_level(confidence):
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "HIGH"
    if confidence >= LOW_CONFIDENCE_THRESHOLD:
        return "MODERATE"
    return "LOW"


def find_last_conv_layer(clf, model_key):
    """model_key must be unique per model ("baseline"/"transfer") - the two
    models have completely different architectures and layer names, so a
    single shared cache here would return the wrong layer name for the
    second model.
    """
    if model_key in _last_conv_layer_cache:
        return _last_conv_layer_cache[model_key]

    for layer in reversed(clf.layers):
        try:
            if len(layer.output.shape) == 4:
                _last_conv_layer_cache[model_key] = layer.name
                return layer.name
        except AttributeError:
            continue
    raise ValueError("Could not find a convolutional layer for Grad-CAM")


def make_gradcam_heatmap(img_array, clf, model_key):
    layer_name = find_last_conv_layer(clf, model_key)
    grad_model = tf.keras.models.Model(
        clf.inputs, [clf.get_layer(layer_name).output, clf.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_img_pil, heatmap, alpha=0.45, resample=None):
    if resample is None:
        resample = Image.Resampling.NEAREST
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(original_img_pil.size, resample=resample)
    heatmap_arr = np.array(heatmap_img) / 255.0

    colored = cm.jet(heatmap_arr)[:, :, :3]
    colored = np.uint8(colored * 255)

    original_arr = np.array(original_img_pil.convert("RGB"))
    superimposed = (colored * alpha + original_arr * (1 - alpha)).astype(np.uint8)

    return Image.fromarray(superimposed)


def make_occlusion_heatmap(img_array, clf, predicted_idx, grid_size=8, occlusion_value=0.5):
    """Occlusion sensitivity mapping - a second, INDEPENDENT explainability
    method from Grad-CAM (no gradients involved at all). Systematically
    covers each patch of a grid_size x grid_size grid with a neutral gray
    value, and measures how much the model's confidence in its own
    predicted class drops for each occlusion. A bigger drop means that
    patch mattered more to the decision. All grid_size*grid_size occluded
    variants are batched into a single predict() call for speed.

    Used to cross-check Grad-CAM: if both methods highlight the same
    region, that's stronger evidence the model's attention is genuine
    rather than an artifact of one particular technique.
    """
    img = img_array[0]
    h, w, _ = img.shape
    patch_h, patch_w = h // grid_size, w // grid_size

    batch = []
    for i in range(grid_size):
        for j in range(grid_size):
            occluded = img.copy()
            occluded[i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w, :] = occlusion_value
            batch.append(occluded)
    batch = np.array(batch)

    occluded_preds = clf.predict(batch, verbose=0)
    occluded_confidences = occluded_preds[:, predicted_idx]

    baseline_confidence = float(clf.predict(img_array, verbose=0)[0][predicted_idx])

    importance = baseline_confidence - occluded_confidences
    importance = np.clip(importance, 0, None)
    if importance.max() > 0:
        importance = importance / importance.max()

    return importance.reshape(grid_size, grid_size)


def encode_image_to_base64(pil_img):
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/performance")
def performance():
    """Model comparison / evaluation dashboard - shows the real results
    from all 5 experiments, and embeds the evaluation plots saved by
    train.py, train_baseline_cnn.py, full_fine_tune.py,
    train_baseline_cnn_v2.py, evaluate_ensemble.py, and
    evaluate_full_metrics.py. If a given plot hasn't been generated yet
    (e.g. you haven't run one of those scripts), that image is simply
    skipped rather than breaking the page.
    """
    models_summary = [
        {"name": "Baseline CNN (from scratch)", "accuracy": "87.25%", "notes": "Best individual model"},
        {"name": "Transfer Learning \u2013 MobileNetV2 (frozen base)", "accuracy": "81.12%", "notes": "Feature extraction only"},
        {"name": "Transfer Learning \u2013 MobileNetV2 (fully fine-tuned)", "accuracy": "83.75%", "notes": "All layers unfrozen"},
        {"name": "Baseline CNN v2 (CLAHE + augmentation + batch norm)", "accuracy": "77.38%", "notes": "Underperformed \u2013 documented negative result"},
        {"name": "Ensemble (Baseline + Fine-tuned Transfer)", "accuracy": "88.06%", "notes": "Final deployed model"},
    ]

    # Only reference images that actually exist in the outputs/ folder, so
    # the page doesn't show broken image icons for scripts you haven't run.
    candidate_images = [
        ("ensemble_confusion_matrix.png", "Confusion Matrix \u2013 Ensemble"),
        ("ensemble_roc_curves.png", "ROC Curves \u2013 Ensemble"),
        ("ensemble_pr_curves.png", "Precision-Recall Curves \u2013 Ensemble"),
        ("accuracy_loss_curves.png", "Training Curves \u2013 Transfer Learning (frozen base)"),
        ("full_finetune_accuracy_loss_curves.png", "Training Curves \u2013 Fully Fine-tuned Transfer Model"),
        ("baseline_v2_accuracy_loss_curves.png", "Training Curves \u2013 Baseline CNN v2 (CLAHE)"),
        ("confusion_matrix.png", "Confusion Matrix \u2013 Baseline CNN"),
        ("baseline_v2_confusion_matrix.png", "Confusion Matrix \u2013 Baseline CNN v2 (CLAHE)"),
    ]
    available_images = [
        {"filename": f, "label": label}
        for f, label in candidate_images
        if os.path.exists(os.path.join(OUTPUTS_DIR, f))
    ]

    return render_template("performance.html", models=models_summary, images=available_images)


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    """Serves evaluation plot images directly from your local outputs/
    folder, so the Model Performance page can display them without
    copying files around."""
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Please upload a PNG or JPG image"}), 400

    try:
        models = load_models_once()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    try:
        file_bytes = file.read()
        input_array, resized_img = preprocess_image(file_bytes)

        # ---- Image validity gate ----
        is_valid, rejection_reason = check_image_validity(resized_img)
        if not is_valid:
            return jsonify({"error": rejection_reason, "rejected_input": True}), 400

        # ---- Run both models and ensemble their predictions ----
        baseline_preds = models["baseline"].predict(input_array, verbose=0)[0]
        transfer_preds = models["transfer"].predict(input_array, verbose=0)[0]
        ensemble_preds = (baseline_preds + transfer_preds) / 2.0

        predicted_idx = int(np.argmax(ensemble_preds))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(ensemble_preds[predicted_idx]) * 100

        probabilities = {
            DISPLAY_NAMES[CLASS_NAMES[i]]: round(float(ensemble_preds[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        # Do the two models even agree? A disagreement is itself a useful
        # reliability signal, separate from raw confidence.
        baseline_top = CLASS_NAMES[int(np.argmax(baseline_preds))]
        transfer_top = CLASS_NAMES[int(np.argmax(transfer_preds))]
        models_agree = baseline_top == transfer_top

        response = {
            "predicted_class": DISPLAY_NAMES[predicted_class],
            "confidence": round(confidence, 2),
            "confidence_level": confidence_level(confidence),
            "probabilities": probabilities,
            "is_tumor": predicted_class != "notumor",
            "low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
            "models_agree": models_agree,
            "model_breakdown": {
                "baseline_cnn": DISPLAY_NAMES[baseline_top],
                "transfer_learning": DISPLAY_NAMES[transfer_top],
            },
        }

        # Grad-CAM from BOTH models - if either fails, the core prediction
        # should still reach the user, so each is isolated in its own
        # try/except.
        try:
            baseline_heatmap = make_gradcam_heatmap(input_array, models["baseline"], "baseline")
            baseline_overlay = overlay_gradcam(resized_img, baseline_heatmap)
            response["gradcam_image_baseline"] = encode_image_to_base64(baseline_overlay)
        except Exception as gradcam_error:
            print(f"Baseline Grad-CAM generation failed (prediction still succeeded): {gradcam_error}")

        try:
            transfer_heatmap = make_gradcam_heatmap(input_array, models["transfer"], "transfer")
            transfer_overlay = overlay_gradcam(resized_img, transfer_heatmap)
            response["gradcam_image_transfer"] = encode_image_to_base64(transfer_overlay)
        except Exception as gradcam_error:
            print(f"Transfer model Grad-CAM generation failed (prediction still succeeded): {gradcam_error}")

        # Occlusion sensitivity - a second, independent explainability method
        # (baseline model only, to keep response time reasonable). Cross-
        # checks Grad-CAM: agreement between the two is stronger evidence
        # the model's attention is genuine.
        try:
            baseline_idx = int(np.argmax(baseline_preds))
            occlusion_map = make_occlusion_heatmap(input_array, models["baseline"], baseline_idx)
            occlusion_overlay = overlay_gradcam(
                resized_img, occlusion_map, resample=Image.Resampling.BILINEAR
            )
            response["occlusion_image_baseline"] = encode_image_to_base64(occlusion_overlay)
        except Exception as occlusion_error:
            print(f"Occlusion sensitivity generation failed (prediction still succeeded): {occlusion_error}")

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
