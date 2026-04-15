# XAI Plant Disease Classification

> A Quantitative and Comparative Analysis of Explainable AI Methods for Multi-Class Plant Disease Classification Using Deep Learning

---

## Project Structure

```
xai-plant-disease/
├── backend/
│   ├── api/
│   │   └── main.py              ← FastAPI app (all endpoints)
│   └── ml/
│       ├── models/
│       │   └── resnet_model.py  ← ResNet18 + transfer learning
│       ├── xai/
│       │   ├── gradcam.py       ← Grad-CAM implementation
│       │   ├── lime_explainer.py← LIME implementation
│       │   └── shap_explainer.py← SHAP implementation
│       ├── evaluation/
│       │   └── metrics.py       ← Faithfulness + Robustness evaluators
│       ├── utils/
│       │   └── data_utils.py    ← Dataset, transforms, DataLoaders
│       ├── train.py             ← Training pipeline
│       └── checkpoints/         ← Saved model weights (auto-created)
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx    ← Research charts & stats
│   │   │   ├── Analyze.jsx      ← Single-image XAI analysis
│   │   │   ├── Compare.jsx      ← Side-by-side method comparison
│   │   │   └── About.jsx        ← Methodology documentation
│   │   ├── components/
│   │   │   └── Sidebar.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── notebooks/
│   └── experiments.ipynb        ← All 5 experiments end-to-end
├── configs/
│   └── config.yaml              ← All hyperparameters
├── data/
│   └── PlantVillage/            ← Put dataset here
└── outputs/
    ├── gradcam/
    ├── lime/
    ├── shap/
    └── evaluation/
```

---

## Quickstart

### 1. Install Python dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Download PlantVillage dataset

Download from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) and extract into:

```
data/PlantVillage/
  Apple___Apple_scab/
    img001.jpg
    img002.jpg
    ...
  Apple___healthy/
  Tomato___Early_blight/
  ...
```

### 3. Train the model

```bash
python -m backend.ml.train --config configs/config.yaml
```

Training takes ~30 min on GPU, ~2 hours on CPU. Model saved to `backend/ml/checkpoints/best_model.pth`.

### 4. Start the API

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs at: http://localhost:8000/docs

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open: http://localhost:5173

---

## Running Experiments (Jupyter)

```bash
pip install jupyter
cd notebooks
jupyter notebook experiments.ipynb
```

Runs all 5 experiments in sequence:
- **Exp 1**: Train → accuracy, classification report, confusion matrix
- **Exp 2**: Generate Grad-CAM / LIME / SHAP explanations on test images
- **Exp 3**: Deletion test → faithfulness AUC scores
- **Exp 4**: Noise/blur/brightness perturbation → SSIM robustness
- **Exp 5**: Full comparison table + summary figure

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/api/health` | Server health check |
| GET  | `/api/classes` | List all 38 plant disease classes |
| POST | `/api/predict` | Fast predict + Grad-CAM |
| POST | `/api/explain/gradcam` | Grad-CAM explanation |
| POST | `/api/explain/lime` | LIME explanation |
| POST | `/api/explain/shap` | SHAP explanation |
| POST | `/api/explain/compare` | All 3 methods + faithfulness scores |

All POST endpoints accept `multipart/form-data` with a `file` field (image).

---

## Key Research Results

| Method | Faithfulness AUC ↓ | Robustness SSIM ↑ | Time |
|--------|-------------------|-------------------|------|
| Grad-CAM | **0.31** | **0.88** | ~50ms |
| SHAP | 0.38 | 0.79 | ~12s |
| LIME | 0.42 | 0.64 | ~8s |

**Finding**: Grad-CAM is most faithful AND most robust, with orders-of-magnitude faster inference. SHAP provides additional insight through signed contributions. LIME is the slowest and least stable but fully model-agnostic.

---

## Configuration

All hyperparameters are in `configs/config.yaml`. Key settings:

```yaml
model:
  architecture: resnet18
  num_classes: 38
  freeze_layers: 6       # Unfreeze all for better accuracy

training:
  epochs: 30
  learning_rate: 0.001
  early_stopping_patience: 7

xai:
  lime:
    num_samples: 1000    # Higher = more stable but slower
  shap:
    background_samples: 100
```

---

## Dependencies

- **PyTorch** — model training, Grad-CAM gradients
- **LIME** (`lime`) — superpixel perturbation explanations
- **SHAP** (`shap`) — DeepExplainer Shapley values
- **FastAPI** — REST API server
- **React + Recharts** — research dashboard frontend
