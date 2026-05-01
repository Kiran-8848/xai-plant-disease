# 🌿 XAI Plant Disease Classification

> A Quantitative and Comparative Analysis of Explainable AI Methods for Multi-Class Plant Disease Classification Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-18+-blue?style=flat-square&logo=react)

---

## 📌 Overview

This project presents a complete deep learning pipeline for plant disease classification combined with a systematic quantitative evaluation of Explainable AI (XAI) techniques. A **ResNet18** model trained via transfer learning on the **PlantVillage dataset** achieves **96.93% test accuracy** across 16 disease classes. Two XAI methods — **Grad-CAM** and **LIME** — are rigorously compared using faithfulness deletion tests, robustness perturbation analysis, and class-wise explanation evaluation.

The project includes:
- ✅ Complete ML training pipeline (PyTorch)
- ✅ Grad-CAM and LIME explanation generation
- ✅ Quantitative faithfulness and robustness evaluation
- ✅ FastAPI REST backend with 6 endpoints
- ✅ Interactive React research dashboard
- ✅ IEEE-format research paper

---

## 🏆 Key Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **96.93%** |
| Macro F1-Score | **0.97** |
| Best Val Accuracy | **97.45%** (Epoch 6) |
| Total Classes | 16 |
| Total Images | 20,632 |
| Training Epochs | 13 (early stopping) |

### XAI Comparison

| Method | Faithfulness AUC ↓ | Robustness SSIM ↑ | Speed |
|--------|-------------------|-------------------|-------|
| **Grad-CAM** | **0.31** | **0.88** | **~0.05s** |
| LIME | 0.42 | 0.64 | ~8.2s |

> **Grad-CAM is 26.2% more faithful, 37.5% more robust, and 164× faster than LIME**

---

## 📁 Project Structure

```
xai-plant-disease/
├── backend/
│   ├── api/
│   │   └── main.py                  ← FastAPI app (6 endpoints)
│   └── ml/
│       ├── models/
│       │   └── resnet_model.py      ← ResNet18 + transfer learning
│       ├── xai/
│       │   ├── gradcam.py           ← Grad-CAM implementation
│       │   └── lime_explainer.py    ← LIME implementation
│       ├── evaluation/
│       │   ├── metrics.py           ← Faithfulness + Robustness
│       │   ├── classwise_analysis.py← Class-wise XAI analysis
│       │   └── visualisations.py   ← Publication figures
│       ├── utils/
│       │   └── data_utils.py        ← Dataset + DataLoaders
│       ├── train.py                 ← Training pipeline
│       ├── evaluate.py              ← Test set evaluation
│       ├── run_xai.py               ← All XAI experiments
│       └── checkpoints/
│           └── class_mapping.json   ← Class name mapping
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx        ← Research charts
│   │   │   ├── Analyze.jsx          ← Single image XAI
│   │   │   ├── Compare.jsx          ← Side-by-side comparison
│   │   │   └── About.jsx            ← Methodology docs
│   │   ├── components/
│   │   │   └── Sidebar.jsx          ← Navigation
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css                ← Dark theme design system
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── notebooks/
│   ├── experiments.ipynb            ← All 5 experiments
│   └── classwise_analysis.ipynb    ← Experiment 6 class analysis
├── configs/
│   └── config.yaml                  ← All hyperparameters
├── data/
│   └── PlantVillage/                ← Dataset goes here
├── outputs/
│   ├── gradcam/                     ← Grad-CAM visualisations
│   ├── lime/                        ← LIME visualisations
│   └── evaluation/                  ← Metrics and figures
├── requirements.txt
└── README.md
```

---

## 🗂️ Dataset

**PlantVillage Dataset** — Hughes & Salathé (2015)

| Detail | Value |
|--------|-------|
| Total Images | 20,632 |
| Classes | 16 |
| Plants | Pepper, Potato, Tomato |
| Image Size | 224 × 224 (resized) |
| Split | 70% / 15% / 15% |

### 16 Classes

| Plant | Disease Classes |
|-------|----------------|
| Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Mosaic Virus, Septoria Leaf Spot, Spider Mites, Target Spot, YellowLeaf Curl Virus, Healthy |

**Download Dataset:**
- Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease
- GitHub: https://github.com/spMohanty/PlantVillage-Dataset

**Citation:**
```
D. P. Hughes and M. Salathé, "An open access repository of images on plant 
health to enable the development of mobile disease diagnostics," 
arXiv preprint arXiv:1511.08060, 2015.
```

---

## 🧠 Model

**ResNet18** with ImageNet Transfer Learning

| Component | Detail |
|-----------|--------|
| Architecture | ResNet18 (He et al., 2016) |
| Pre-training | ImageNet (1.28M images) |
| Parameters | ~11.18 Million |
| Custom Head | Dropout(0.3) + Linear(512 → 16) |
| Optimizer | Adam (lr=0.001, wd=0.0001) |
| Scheduler | Cosine Annealing |
| Loss | Cross-Entropy |
| Batch Size | 32 |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- NVIDIA GPU (recommended) or CPU

### 1. Clone Repository
```bash
git clone https://github.com/your-username/xai-plant-disease.git
cd xai-plant-disease
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### 5. Setup Dataset
Download PlantVillage from Kaggle and place in:
```
data/
└── PlantVillage/
    ├── Pepper__bell___Bacterial_spot/
    ├── Pepper__bell___healthy/
    ├── Potato___Early_blight/
    ├── Potato___Late_blight/
    ├── Potato___healthy/
    ├── Tomato___Bacterial_spot/
    ├── Tomato___Early_blight/
    └── ... (all 16 class folders)
```

### 6. Update Config
Open `configs/config.yaml`:
```yaml
data:
  root: "./data/PlantVillage"
  num_classes: 16
  num_workers: 0        # Windows users keep this 0

model:
  num_classes: 16
```

---

## 🚀 Usage

### Step 1 — Verify Dataset
```bash
python -c "
from pathlib import Path
root = Path('data/PlantVillage')
classes = [d.name for d in sorted(root.iterdir()) if d.is_dir()]
print(f'Found {len(classes)} classes:')
for c in classes:
    count = len(list((root / c).iterdir()))
    print(f'  {c}  ->  {count} images')
"
```

### Step 2 — Train Model
```bash
mkdir backend\ml\checkpoints     # Windows
# mkdir -p backend/ml/checkpoints  # Mac/Linux

python -m backend.ml.train --config configs/config.yaml
```

Expected output:
```
[Device] GPU: NVIDIA GeForce RTX 3050 Laptop GPU
[Data] Train: 14446 | Val: 3095 | Test: 3097
[Data] Classes: 16

Epoch 001/30 | Train Loss: 0.4574 Acc: 0.8549 | Val Loss: 0.2267 Acc: 0.9257
Epoch 006/30 | Train Loss: 0.1367 Acc: 0.9556 | Val Loss: 0.0787 Acc: 0.9745 ← best
...
[Early Stop] No improvement for 7 epochs.
[Done] Best Val Accuracy: 0.9745
```

### Step 3 — Evaluate Model
```bash
python -m backend.ml.evaluate
```

Expected output:
```
Test Accuracy: 96.93%

              precision  recall  f1-score
Tomato_healthy    1.00    0.98    0.99
...
macro avg         0.96    0.97    0.97
```

### Step 4 — Run XAI Experiments
```bash
python -m backend.ml.run_xai
```

Generates:
- `outputs/gradcam/gradcam_grid.png`
- `outputs/lime/lime_grid.png`
- `outputs/evaluation/faithfulness_gradcam.png`
- `outputs/evaluation/xai_comparison.png`
- `outputs/evaluation/final_results.json`

### Step 5 — Start API Server
```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: `http://localhost:8000/docs`

### Step 6 — Start Frontend
```bash
cd frontend
npm run dev
```

Open: `http://localhost:5173`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Server health check |
| GET | `/api/classes` | List all 16 class names |
| POST | `/api/predict` | Classify image + Grad-CAM |
| POST | `/api/explain/gradcam` | Grad-CAM explanation |
| POST | `/api/explain/lime` | LIME explanation |
| POST | `/api/explain/compare` | Both methods + faithfulness |

All POST endpoints accept `multipart/form-data` with a `file` field (image).

### Example API Call
```python
import requests

with open("leaf.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/explain/gradcam",
        files={"file": f}
    )

result = response.json()
print(f"Prediction: {result['pred_label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 📊 Detailed Results

### Per-Class F1 Scores

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Pepper Bell Bacterial Spot | 0.93 | 0.97 | 0.95 | 144 |
| Pepper Bell Healthy | 0.97 | 0.96 | 0.96 | 223 |
| Potato Early Blight | 0.99 | 0.99 | 0.99 | 146 |
| Potato Healthy | 0.93 | 1.00 | 0.96 | 27 |
| Potato Late Blight | 0.96 | 0.98 | 0.97 | 160 |
| Tomato Bacterial Spot | 0.98 | 0.98 | 0.98 | 315 |
| Tomato Early Blight | 0.84 | 0.94 | **0.89** | 145 |
| Tomato Healthy | 1.00 | 0.98 | 0.99 | 234 |
| Tomato Late Blight | 0.95 | 0.93 | 0.94 | 273 |
| Tomato Leaf Mold | 1.00 | 0.93 | 0.96 | 131 |
| Tomato Mosaic Virus | 0.93 | 1.00 | 0.97 | 70 |
| Tomato Septoria Leaf Spot | 0.98 | 0.98 | 0.98 | 256 |
| Tomato Spider Mites | 0.97 | 0.97 | 0.97 | 242 |
| Tomato Target Spot | 0.98 | 0.89 | 0.94 | 219 |
| Tomato YellowLeaf Curl Virus | 0.98 | 1.00 | **0.99** | 512 |
| **Macro Average** | **0.96** | **0.97** | **0.97** | **3,097** |

### Faithfulness Results (Deletion Test)

| Method | AUC ↓ | @10% Drop | @30% Drop | @50% Drop |
|--------|-------|-----------|-----------|-----------|
| Grad-CAM | **0.31** | **35.2%** | **65.1%** | **77.3%** |
| LIME | 0.42 | 18.7% | 40.9% | 58.0% |

### Robustness Results (SSIM under Gaussian Noise)

| Method | σ=0.05 | σ=0.10 | σ=0.20 | Mean |
|--------|--------|--------|--------|------|
| Grad-CAM | **0.91** | **0.87** | **0.81** | **0.88** |
| LIME | 0.72 | 0.63 | 0.54 | 0.64 |

### Class-wise Explanation Statistics

| Category | Coverage ↓ | Consistency ↑ |
|----------|-----------|--------------|
| Diseased (13 classes) | **0.21 ± 0.08** | **0.74 ± 0.09** |
| Healthy (3 classes) | 0.38 ± 0.11 | 0.61 ± 0.12 |

---

## 🔬 XAI Methods

### Grad-CAM
- **Type:** Gradient-based
- **Target Layer:** conv2 of last BasicBlock in ResNet18 Layer4
- **Formula:** `L_GradCAM = ReLU(Σk αkc · Ak)` where `αkc = (1/Z) Σij (∂yc/∂Akij)`
- **Output:** Pixel-level heatmap upsampled to 224×224
- **Time:** ~0.050s per image

### LIME
- **Type:** Perturbation-based (Model-agnostic)
- **Segmentation:** Quickshift (kernel=4, max_dist=200, ratio=0.2)
- **Samples:** 500 perturbations per image
- **Features:** Top 10 superpixels highlighted
- **Formula:** `ξ(x) = argmin L(f, g, πx) + Ω(g)`
- **Time:** ~8.2s per image

---

## 🖥️ Frontend Pages

| Page | Description |
|------|-------------|
| **Dashboard** | Training curves, faithfulness charts, radar chart, per-class F1, summary table |
| **Analyze Image** | Upload leaf image, select Grad-CAM or LIME, view explanation + top-5 predictions |
| **Compare XAI** | Run both methods on same image, side-by-side results, faithfulness curve chart |
| **Methodology** | Research documentation, XAI formulas, evaluation metrics, contributions |

---

## 📓 Jupyter Notebooks

```bash
pip install jupyter
jupyter notebook
```

| Notebook | Experiments |
|----------|-------------|
| `experiments.ipynb` | Exp 1-5: Training, XAI generation, faithfulness, robustness, comparison |
| `classwise_analysis.ipynb` | Exp 6: Healthy vs diseased analysis, similarity matrix |

---

## 🛠️ Tech Stack

### Backend
- **PyTorch 2.0** — Model training and inference
- **torchvision** — ResNet18 and transforms
- **FastAPI** — REST API framework
- **LIME** — Local Interpretable Model-Agnostic Explanations
- **OpenCV** — Image processing and heatmap overlay
- **scikit-learn** — Evaluation metrics

### Frontend
- **React 18** — UI framework
- **Vite** — Build tool and dev server
- **Recharts** — Charts and visualisations
- **React Dropzone** — Image upload
- **Axios** — HTTP client
- **Lucide React** — Icons

---

## ⚠️ Common Issues

### Windows — MultiProcessing Error
```yaml
# configs/config.yaml
data:
  num_workers: 0    # Set to 0 on Windows
```

### CUDA Out of Memory
```yaml
# configs/config.yaml
data:
  batch_size: 16    # Reduce from 32 to 16
```

### ModuleNotFoundError
```bash
# Always run from project root
cd xai-plant-disease
python -m backend.ml.train    # use -m flag
```

### Port Already in Use
```bash
uvicorn backend.api.main:app --port 8001    # use different port
```
---

## ⭐ If this project helped you, please give it a star ⭐️ !
