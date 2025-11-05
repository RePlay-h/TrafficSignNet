# 🚦 TrafficSignNet
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![DVC](https://img.shields.io/badge/DVC-945DD6?logo=dvc&logoColor=white)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DagsHub](https://img.shields.io/badge/DagsHub-orange.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABZ0lEQVQ4T63Tu0sDQRjF8U+yRCRVKhVYxWIRaBUsB6qSdp3CIuiJ2AkvoC7ESEIUXkAq2oIK4BVu1YlFVFboZXs8O9OHkZ4yM3uzh773bNnnXgMuMd1VgPRngB4xAhoWw2q3QbmWZ9CAtE9GzBfy8mUfMPsPjOWGQxRhjHvwF1Uwo4D4PZ7u5S0DcoUM4NCDHDVYq4+7SrZ/NE0MjoAVowE1waC5gBi7kC10dQ3EQ0EwxdchGZNoj1wAaUJncspg9IuWRzRJNoTvKqpXrct+JAvDOhgf9YjXkFivhzNeDgDfO7NeJ0D6FoCNUUbCj1d91Qy1FXRcXTRyH20QUQPhsJq0DmZgZqENAhH5R8Zy8wRChPz4l27bbhhctZ/jKhp48qSkAAAABJRU5ErkJggg==)](https://dagshub.com/)

TrafficSignNet is a deep learning project for traffic sign classification using the **GTSRB** dataset.  
It is built with **PyTorch**, **DVC**, and **MLflow**, and integrates seamlessly with **DagsHub** for experiment tracking and data versioning.

---

## 📂 Project structure

```
TrafficSignNet/
├── data/                 # DVC-managed data (raw, processed)
├── models/               # Saved models (.pth)
├── src/
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model definitions (CNN)
│   ├── api/              # FastAPI inference endpoint
│   ├── train.py          # Training script with MLflow logging
│   ├── evaluate.py       # Evaluation and metrics logging
├── dvc.yaml              # DVC pipeline definition
├── params.yaml           # Parameters for preprocessing and training
├── requirements.txt      # Dependencies
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/RePlay-h/TrafficSignNet.git
cd TrafficSignNet
```

### 2️⃣ Set up the environment
We recommend using **Anaconda**:
```bash
conda create -n trafficnet python=3.10
conda activate trafficnet
pip install -r requirements.txt
```

---

## 🚀 Training

Run the full DVC pipeline:
```bash
dvc repro
```

Or manually train the model:
```bash
python src/train.py
```

Trained models are saved in `models/` and automatically tracked by **MLflow** and **DVC**.

---

## 📊 Experiment tracking

This project uses **MLflow** integrated with **DagsHub**.  
All metrics, parameters, and artifacts are automatically synced to your repository at:

👉 [https://dagshub.com/RePlay-h/TrafficSignNet.mlflow](https://dagshub.com/RePlay-h/TrafficSignNet.mlflow)

---

## 🧪 Evaluation

To evaluate the best model:
```bash
python src/evaluate.py
```

Results and accuracy are logged to MLflow.

---

## 🌐 API Deployment

You can serve the trained model via FastAPI:

```bash
python -m src.api.main
```

Example request:
```bash
POST /predict
{
  "image": "base64_encoded_image_here"
}
```

Response:
```json
{
  "class_name": "Stop Sign"
}
```

---

## 🧰 Tools used

- **PyTorch** — model training  
- **Albumentations** — data augmentation  
- **DVC** — data and pipeline versioning  
- **MLflow** — experiment tracking  
- **DagsHub** — remote storage and visualization  
- **FastAPI** — inference endpoint  

---

## 📈 Future work
- Improve model accuracy  
- Add ONNX export for deployment  
- Create Streamlit demo  

---

