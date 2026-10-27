# NeuroScan: Explainable and Uncertainty-Aware Brain Tumor MRI Classification

B.Tech 4-1 Mini Project, Department of Artificial Intelligence, Anurag University.
Team: K. Venkata Rohan (23EG106B30), A. Mohan Sujith (23EG106B04), M. Mahathi (23EG106B41).

The project classifies a brain MRI slice into **glioma, meningioma, pituitary tumor or no tumor**.
Five training strategies are compared on one test set, and the best system (an ensemble) is
packaged as a Flask web application with Grad-CAM and occlusion-sensitivity explanations and a
reliability check (model agreement, confidence threshold, input gate).

> This is an academic prototype. It has not been clinically validated and must not be used for diagnosis.

## Results (1,600-image test set)

| Experiment | Model | Test accuracy |
|---|---|---|
| 1 | MobileNetV2, frozen base | 81.12% |
| 2 | CNN trained from scratch | 87.25% |
| 3 | MobileNetV2, fully fine-tuned | 83.75% |
| 4 | CNN + CLAHE + batch normalization | 77.38% |
| 5 | Ensemble of Experiments 2 and 3 | **88.06%** (macro ROC-AUC 0.976) |

On an independent dataset (BDNeuro-MRI, 889 images, no retraining) the ensemble scored 80.99%.

## Folder layout

```
app.py                         Flask backend (two models, Grad-CAM, occlusion, reliability check)
train.py                       Experiment 1: MobileNetV2, frozen base
train_baseline_cnn.py          Experiment 2: CNN from scratch
full_fine_tune.py              Experiment 3: MobileNetV2, all layers fine-tuned
train_baseline_cnn_v2.py       Experiment 4: CNN + CLAHE + batch norm
evaluate_ensemble.py           Experiment 5: ensemble evaluation
evaluate_full_metrics.py       ROC-AUC, PR curves, sensitivity/specificity
evaluate_cross_dataset.py      Independent-dataset evaluation (one model)
evaluate_cross_dataset_all.py  Independent-dataset evaluation (all models and the ensemble)
export_weights.py              Saves weights-only files for Colab
templates/, static/            Web interface
outputs/                       Curves, confusion matrices and reports saved by the scripts
Dockerfile, requirements.txt   Packaging
```

The trained models (`*.h5`) are in this repository. The application needs
`baseline_cnn_model.h5` and `transfer_model_full_finetuned.h5` next to `app.py`.

## 1. Set up

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux
pip install -r requirements.txt
```

Versions used by the team (from `pip freeze`): TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.5.3,
scikit-learn 1.9.1, Matplotlib 3.11.2, seaborn 0.13.2, opencv-python 5.0.0.93, Flask 3.1.3, Pillow 12.3.0.
`requirements.txt` does not pin versions, so a fresh install may pick newer releases.

## 2. Get the dataset

Main dataset: **Brain Tumor MRI Dataset** by Masoud Nickparvar (Kaggle):
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Check the licence shown on that page before reuse.

Download it, unzip it, and place the `Training` and `Testing` folders inside a `data` folder:

```
data/
  Training/   glioma, meningioma, notumor, pituitary
  Testing/    glioma, meningioma, notumor, pituitary
```

The class folders are read in alphabetical order. This order must match `CLASS_NAMES` in `app.py`.

The independent test set is BDNeuro-MRI (Mendeley Data):
https://data.mendeley.com/datasets/zwr4ntf94j

## 3. Reproduce the experiments

No random seed is fixed, so results will be close to, but not identical with, the numbers above.
Each run takes from about ten minutes to over an hour on a CPU.

```bash
python train.py                      # Exp. 1  -> brain_tumor_model.h5
python train_baseline_cnn.py         # Exp. 2  -> baseline_cnn_model.h5
python full_fine_tune.py             # Exp. 3  -> transfer_model_full_finetuned.h5 (needs Exp. 1)
python train_baseline_cnn_v2.py      # Exp. 4  -> baseline_cnn_v2_model.h5
python evaluate_ensemble.py          # Exp. 5  ensemble accuracy, report, confusion matrix
python evaluate_full_metrics.py      # ROC-AUC, PR curves, sensitivity/specificity
python evaluate_cross_dataset_all.py --data_dir path/to/BDNeuro-MRI/test
```

Common options: `--data_dir`, `--epochs`, `--batch_size`, `--output_dir` (default `outputs`).
Training curves, confusion matrices and classification reports are saved in `outputs/`.

## 4. Run the web application

```bash
python app.py
```

Open http://localhost:5000, upload an MRI slice and press Analyze. The page shows the predicted
class, the ensemble confidence, a confidence badge (HIGH, MODERATE or LOW), the Grad-CAM maps,
the occlusion map and a warning when the result is unreliable.

### Docker

```bash
docker build -t neuroscan .
docker run -p 7860:7860 neuroscan
```

The container uses gunicorn with a 120-second timeout, because one request runs two models,
two Grad-CAM computations and 64 occlusion predictions.

## Reused components

MobileNetV2 and its ImageNet weights (Keras applications), the public datasets above, TensorFlow/Keras,
Flask, OpenCV and NumPy. Grad-CAM and occlusion sensitivity are published methods implemented by the team.
The experiments, scripts, ensemble, reliability layer, input gate and web application are the team's own work.
