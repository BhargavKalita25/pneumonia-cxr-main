# Pneumonia Detection from Chest X-Rays (DenseNet-121, PyTorch)

⚠️ **For research & education only – not for clinical use.**

This project provides a complete deep learning pipeline for pneumonia detection from chest X-ray (CXR) images.  
It includes training, evaluation, explainability (Grad-CAM), a FastAPI backend, and a Streamlit UI.

---

## 📥 Clone the Repository

```powershell
git clone https://github.com/BhargavKalita25/pneumonia-cxr.git
cd pneumonia-cxr
```

---

## 📂 Dataset Setup

We use the **Kaggle Chest X-Ray Images (Pneumonia)** dataset:  
🔗 [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

After downloading, extract and place it in:

```
pneumonia-cxr/
  data/kaggle_chest_xray/
    train/NORMAL/...
    train/PNEUMONIA/...
    val/NORMAL/...
    val/PNEUMONIA/...
    test/NORMAL/...
    test/PNEUMONIA/...
```

---

## ⚙️ Environment Setup

### 1. Create and activate a virtual environment

**Windows (PowerShell / CMD):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux (bash/zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Training

Run training:

```bash
python -m app.train
```

- Default: **DenseNet-121**, 20 epochs, mixed precision (AMP) if GPU available.  
- Profiles (`app/profiles/*.yaml`) adjust batch size & epochs for T4, V100, or CPU.

---

## 📊 Evaluation

Run evaluation on the test set:

```bash
python -m app.evaluate
```

Outputs saved in `outputs/`:
- `test_metrics.json` with Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC  
- ROC and PR curves  
- Calibration plot  
- Confusion matrix  

---

## 🌐 API Server

Start the FastAPI backend:

```bash
uvicorn app.service.api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health` → service status  
- `GET /models` → available models  
- `POST /predict` → predict on one image  
- `POST /predict-batch` → batch inference  

Docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 💻 Streamlit UI

In a **second terminal** (keep the API running):

```bash
streamlit run ui/streamlit_app.py
```

Features:
- Drag & drop image upload (PNG/JPG, optional DICOM)  
- Prediction label + confidence  
- Grad-CAM overlay toggle + download option  
- Batch inference with history panel  
- Metrics dashboard (loss/accuracy, ROC, PR, calibration)  

---

## 📊 Results & Visualization

- **Training curves**: `outputs/loss.png`, `outputs/acc.png`  
- **ROC/PR curves**: `outputs/roc.png`, `outputs/pr.png`  
- **Calibration plot**: `outputs/calibration.png`  
- **Grad-CAM overlays**: Saved automatically during inference and displayed in Streamlit  

---

## 🎯 Project Targets

- ROC-AUC ≥ 0.90 on held-out test set  
- Sensitivity @ 90% specificity and vice versa  
- Grad-CAM highlights plausible lung regions  
- UI loads in <2s, inference <3s on GPU  

---

## ⚖️ Notes

- Dataset is pediatric; may not generalize to adults.  
- Folder-based labels can include some noise.  
- No PHI is collected or stored.  
- Disclaimer shown: *Not for diagnostic use*.  

---

## 📜 License

MIT




