# Brain Tumor Detection — VS Code Project

Everything for the mini project in one folder: training script, trained
model output, and a Flask + HTML/CSS/JS web app to demo it.

```
brain_tumor_project/
├── README.md
├── requirements.txt
├── train.py                # trains the model, run this first
├── app.py                  # Flask backend, run this after training
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── data/                   # put the dataset here (see step 3 below)
└── outputs/                # training saves plots + report here
```

## 1. Open the project in VS Code

- File → Open Folder → select `brain_tumor_project`
- Install the **Python extension** (ms-python.python) from the Extensions
  panel if you don't have it already — VS Code will usually prompt you to
  install it automatically when it sees `.py` files.

## 2. Set up a virtual environment

Open a terminal in VS Code (`` Ctrl+` ``, or Terminal → New Terminal) and run:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Then install everything:
```bash
pip install -r requirements.txt
```

**Select the interpreter:** press `Ctrl+Shift+P` → "Python: Select Interpreter"
→ choose the one inside `venv` (it'll be listed as recommended). This makes
sure VS Code runs your code with the right packages installed.

## 3. Get the dataset

Kaggle: **"Brain Tumor Classification (MRI)"** by Sartaj Bhuvaji.

**Option A — Kaggle API:**
```bash
pip install kaggle
```
Get an API token from kaggle.com → Account → Create New API Token (downloads
`kaggle.json`). Put it at:
- Windows: `C:\Users\<you>\.kaggle\kaggle.json`
- Mac/Linux: `~/.kaggle/kaggle.json`

Then:
```bash
kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri
```
Unzip it so you end up with `data/Training/` and `data/Testing/` inside this
project folder.

**Option B — Manual:** download the zip from the Kaggle page yourself and
extract it into `data/`, same target structure as above.

## 4. Train the model

In the VS Code terminal (with `venv` active):
```bash
python train.py
```

This trains a MobileNetV2 transfer-learning model, prints progress per
epoch, and when done:
- Saves `brain_tumor_model.h5` in the project root (right where `app.py`
  expects it — no moving files around)
- Saves accuracy/loss curves, a confusion matrix, and a classification
  report into `outputs/` — these are your report figures

Optional flags:
```bash
python train.py --epochs 20          # train longer
python train.py --data_dir data      # if your dataset is elsewhere
```

You can also just click the **Run** ▷ button at the top-right of `train.py`
in VS Code instead of typing the command.

**No GPU locally?** It'll still work, just slower (maybe 20–40 min instead
of a few minutes) since MobileNetV2's base is frozen and you're only
training a small head. If that's too slow for your deadline, train in
Google Colab instead (free GPU) and copy the resulting `brain_tumor_model.h5`
into this folder afterwards — everything else works the same either way.

## 5. Run the web app

```bash
python app.py
```
Open **http://localhost:5000** in your browser. Upload an MRI image, click
"Analyze scan", and you'll see the predicted class, confidence, and a
probability breakdown across all four classes.

## Debugging in VS Code (optional)

To step through the code with breakpoints instead of just running it:
- Click in the left margin of a line to set a breakpoint
- Go to Run and Debug (`Ctrl+Shift+D`) → "Run and Debug" → select "Python File"
- This works for both `train.py` and `app.py`

## Important: class order

`app.py` assumes:
```python
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
```
`train.py` prints the actual order it trained with (`Classes found: [...]`)
— double check it matches. If not, edit `CLASS_NAMES` in `app.py`.

## Notes for your report / viva

- **Frontend**: vanilla HTML/CSS/JS, no framework — `fetch()` + `FormData`
  to send the image, renders the JSON response.
- **Backend**: Flask, one page route (`/`) and one API route (`/predict`).
- **Training**: transfer learning with MobileNetV2 (frozen base, ImageNet
  weights) + a small trainable classification head — faster and less
  data-hungry than training a CNN from scratch, which matters given the
  dataset size.
- **Inference flow**: uploaded image → resized to 224×224 → normalized to
  [0,1] → same preprocessing as training → passed through the model →
  softmax output converted to per-class percentages.
- Be ready to explain why the preprocessing in `app.py` has to exactly match
  `train.py` — mismatched preprocessing is a common reason a deployed model
  underperforms compared to evaluation.
- This is a demo tool for an academic project, not a diagnostic device —
  worth stating explicitly if asked in viva.
