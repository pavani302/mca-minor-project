# 🫁 Pneumonia Detection from Chest X-Rays using Deep Learning

An MCA minor project that detects **pneumonia** from chest X-ray images using a
**DenseNet121** convolutional neural network (transfer learning), with
**Grad-CAM** explainability and an interactive **Streamlit** web app.

> ⚠️ **Educational demonstration only — not a medical device.** This project is
> a student minor project and must not be used for real diagnosis.

---

## ✨ Features

- **Deep-learning classifier** — DenseNet121 pretrained on ImageNet, fine-tuned
  in two phases on chest X-rays (Normal vs Pneumonia).
- **Explainable AI** — Grad-CAM heatmaps show which lung regions drove each
  prediction.
- **Proper evaluation** — accuracy, precision, recall, F1, AUC, confusion
  matrix and ROC curve on a held-out test set.
- **Web app with three sections:**
  1. **Try It** — upload an X-ray → prediction + confidence + Grad-CAM.
  2. **How It Works** — a visual explanation of the pipeline, model and results.
  3. **Source Code** — browse the whole project in the browser.

---

## 🗂️ Project structure

```
minorproject/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/            # full/subset dataset (git-ignored, downloaded)
│   └── sample/         # a few example X-rays for the app
├── src/
│   ├── config.py       # paths & hyperparameters (single source of truth)
│   ├── download_data.py# fetch a balanced subset of the dataset
│   ├── data_loader.py  # tf.data pipelines + augmentation
│   ├── model.py        # DenseNet121 transfer-learning model
│   ├── gradcam.py      # Grad-CAM explainability
│   ├── train.py        # two-phase training
│   ├── evaluate.py     # metrics + report figures
│   └── predict.py      # single-image inference (CLI + app)
├── models/             # trained model + metrics.json
├── reports/figures/    # training curves, confusion matrix, ROC, Grad-CAMs
└── app/
    ├── Home.py
    └── pages/
        ├── 1_Try_It.py
        ├── 2_How_It_Works.py
        └── 3_Source_Code.py
```

---

## 🚀 Setup

Requires **Python 3.11**.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Usage

### 1. Download the data

```bash
python -m src.download_data          # small balanced subset (fast baseline)
python -m src.download_data --full   # the full dataset (best accuracy)
```

This pulls a public HuggingFace mirror of the Kaggle *Chest X-Ray Images
(Pneumonia)* dataset — **no Kaggle account needed** — and writes it in an
`ImageFolder` layout under `data/raw/`.

### 2. Train the model

```bash
python -m src.train
```

Two-phase transfer learning (freeze → fine-tune). Saves the model to
`models/pneumonia_densenet121.keras`, metrics to `models/metrics.json`, and
figures to `reports/figures/`.

### 3. Evaluate (optional, standalone)

```bash
python -m src.evaluate
```

### 4. Predict on a single image (CLI)

```bash
python -m src.predict path/to/xray.jpeg
```

### 5. Run the web app

```bash
streamlit run app/Home.py
```

Then open the URL Streamlit prints (usually <http://localhost:8501>).

---

## 🧠 How it works

```
Chest X-ray ─▶ Preprocess ─▶ DenseNet121 ─▶ Sigmoid ─▶ NORMAL / PNEUMONIA
               (224×224 RGB,  (fine-tuned)   (0→1)      + confidence
                normalised)                             + Grad-CAM heatmap
```

- **Transfer learning:** DenseNet121's ImageNet features are reused and adapted
  to X-rays, so we get high accuracy without a massive dataset.
- **Class imbalance:** the dataset has more pneumonia than normal images, so
  inverse-frequency **class weights** are applied during training.
- **Regularisation:** dropout, data augmentation, early stopping and
  learning-rate reduction prevent overfitting.
- **Grad-CAM:** gradients of the pneumonia score w.r.t. the last convolutional
  feature map produce a heatmap over the X-ray, making the decision
  interpretable.

---

## 🛠️ Tech stack

Python · TensorFlow/Keras · OpenCV · NumPy · Pandas · scikit-learn ·
Matplotlib/Seaborn · Streamlit

---

## 📄 Project report

The final report is `Pneumonia_Detection_Final_Report.docx`, generated from
`docs/build_report.js` (it reads the real `models/metrics.json` and the figures
in `reports/figures/`). To regenerate it:

```bash
npm install docx@8.5.0
node docs/build_report.js
```

The Table of Contents is typed out statically (so it displays correctly in
Word and Google Docs without an "update field" step). Page numbers are shown in
the page footers. To fill exact page numbers into the TOC, render once on a
machine with LibreOffice and rebuild:

```bash
# Ubuntu/EC2: sudo apt-get install -y libreoffice
node docs/build_report.js            # 1st pass (clean outline)
python docs/compute_toc_pages.py     # renders + computes page numbers
node docs/build_report.js            # 2nd pass (numbered TOC)
```

Alternatively, in Google Docs choose **Insert ▸ Table of contents** — it builds
a numbered, clickable TOC from the report's heading styles.

---

## 🚀 Deploying on Ubuntu / EC2

```bash
git clone <repository-url> && cd minorproject
chmod +x deploy.sh && ./deploy.sh     # prompts: small sample or full dataset
```

`deploy.sh` installs dependencies, sets up the environment, downloads the data,
trains the model and launches the app on `0.0.0.0:8501`. Remember to open
inbound TCP 8501 in the EC2 security group.

---

## 👤 Author

**Damaraparapu Krishna Pavani** — MCA, Amrita
Minor Project (21CSA697A)

## 📚 Dataset citation

Kermany, D., Zhang, K., Goldbaum, M. (2018). *Labeled Optical Coherence
Tomography (OCT) and Chest X-Ray Images for Classification.* Mendeley Data.
Originally on Kaggle as *Chest X-Ray Images (Pneumonia)*.
