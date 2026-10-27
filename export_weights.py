"""
Exports weights-only files from the trained models, for the results
notebook to load. Weight-loading is far less sensitive to Keras version
differences than loading a full saved model, which requires exactly
reconstructing internal layer configs.

Run this locally where the models already load correctly:
    python export_weights.py
"""

from tensorflow.keras.models import load_model

print("Loading baseline_cnn_model.h5 ...")
baseline = load_model("baseline_cnn_model.h5")
baseline.save_weights("baseline_cnn.weights.h5")
print("Saved baseline_cnn.weights.h5")

print("Loading transfer_model_full_finetuned.h5 ...")
transfer = load_model("transfer_model_full_finetuned.h5")
transfer.save_weights("transfer_model.weights.h5")
print("Saved transfer_model.weights.h5")

print("\nDone. Push these two new .weights.h5 files to GitHub next.")
