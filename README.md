⸻

🌿 XAI Plant Disease Classification

A Quantitative and Comparative Analysis of Explainable AI Methods for Multi-Class Plant Disease Classification Using Deep Learning

⸻

📌 Overview

This project presents a deep learning-based plant disease classification system enhanced with Explainable AI (XAI) techniques to interpret model predictions.

* Model: ResNet18 (Transfer Learning)
* Dataset: PlantVillage Dataset
* Classes: 16 plant disease categories
* XAI Methods: Grad-CAM, LIME
* Framework: PyTorch + FastAPI + React

👉 Focus: Accuracy + Explainability + Trust in AI decisions

⸻

🗂️ Project Structure


              xai-plant-disease/
├── backend/
│   ├── api/
│   │   └── main.py              ← FastAPI app
│   └── ml/
│       ├── models/
│       │   └── resnet_model.py
│       ├── xai/
│       │   ├── gradcam.py
│       │   └── lime_explainer.py
│       ├── evaluation/
│       │   └── metrics.py
│       ├── utils/
│       │   └── data_utils.py
│       ├── train.py
│       └── checkpoints/
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Analyze.jsx
│   │   │   ├── Compare.jsx
│   │   │   └── About.jsx
│   │   ├── components/
│   │   │   └── Sidebar.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
├── notebooks/
│   └── experiments.ipynb
├── configs/
│   └── config.yaml
├── data/
│   └── PlantVillage/
└── outputs/
    ├── gradcam/
    ├── lime/
    └── evaluation/
⸻

⚡ Quickstart

1. Install Dependencies

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

⸻

2. Download Dataset

Download from Kaggle and place in:

data/PlantVillage/

⸻

3. Train Model

python -m backend.ml.train --config configs/config.yaml

* Training Time: ~18–30 min (GPU)
* Model Path: backend/ml/checkpoints/best_model.pth

⸻

4. Run Backend API

uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

Docs: http://localhost:8000/docs

⸻

5. Run Frontend

cd frontend
npm install
npm run dev

Open: http://localhost:5173

⸻

🧪 Experiments

pip install jupyter
cd notebooks
jupyter notebook experiments.ipynb

Experiments Included:

* Exp 1: Model training + confusion matrix
* Exp 2: Grad-CAM & LIME explanations
* Exp 3: Faithfulness (Deletion Test)
* Exp 4: Robustness (Noise → SSIM)
* Exp 5: Comparison analysis

⸻

🔌 API Endpoints

Method	Endpoint	Description
GET	/api/health	Health check
GET	/api/classes	List 16 classes
POST	/api/predict	Prediction + Grad-CAM
POST	/api/explain/gradcam	Grad-CAM
POST	/api/explain/lime	LIME
POST	/api/explain/compare	Compare both

⸻

📊 Key Results

Model Performance

* Test Accuracy: 96.93%
* Macro F1-score: 0.97
* Best Validation Accuracy: 97.45%

⸻

XAI Comparison

Metric	Grad-CAM	LIME
Faithfulness (AUC ↓)	0.31	0.42
Robustness (SSIM ↑)	0.88	0.64
Time	0.05s	8.20s

⸻

🧠 Key Findings

* Grad-CAM is more faithful, robust, and faster
* LIME is model-agnostic but slower and less stable
* Model focuses on disease-specific regions

⸻

⚙️ Configuration

model:
  architecture: resnet18
  num_classes: 16
training:
  epochs: 30
  learning_rate: 0.001
  early_stopping_patience: 7
xai:
  lime:
    num_samples: 500

⸻

🧰 Tech Stack

Backend:

* PyTorch
* FastAPI

Frontend:

* React
* Vite
* Recharts

⸻

💻 Hardware

* GPU: NVIDIA RTX 3050
* Training: ~18 minutes
* Inference: ~0.018 sec
* Grad-CAM: ~0.032 sec
* LIME: ~8.2 sec

⸻

⚠️ Limitations

* Controlled dataset background
* LIME randomness affects stability
* XAI methods are approximate

⸻

🚀 Future Work

* Real-world dataset testing
* EfficientNet comparison
* Faster XAI techniques
* Mobile deployment

⸻

📌 Research Contribution

* Quantitative XAI evaluation
* Faithfulness + robustness analysis
* Grad-CAM vs LIME comparison
* Full-stack interpretable AI system

⸻

🏁 Conclusion

ResNet18 achieved 96.93% accuracy, and Grad-CAM outperformed LIME in faithfulness, robustness, and speed, making it more suitable for real-time interpretable plant disease detection systems.

⸻

⭐ Support

If you like this project, give it a ⭐ on GitHub

:::

⸻
