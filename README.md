# 🛡️ FraudShield - Real-Time Fraud Detection Engine

A standard end-to-end MLOps project that detects fraudulent transactions in real-time using XGBoost and FastAPI. Containerized with Docker.

## 🚀 Key Features
* **Real-Time Inference:** < 100ms latency using FastAPI.
* **Production-Ready Model:** XGBoost Classifier with 0.92 AUC.
* **Imbalance Handling:** Trained using SMOTE techniques.
* **Containerized:** Fully Dockerized for "run anywhere" deployment.
* **Traffic Simulator:** Includes a script to mimic live payment traffic.

## 🛠️ Tech Stack
* **Model:** XGBoost, Scikit-Learn, Pandas
* **API:** FastAPI, Uvicorn, Pydantic
* **DevOps:** Docker
* **Language:** Python 3.10

## 📊 Performance
| Metric | Score |
| :--- | :--- |
| **ROC-AUC** | **0.92** |
| **Precision** | 0.73 |
| **Recall** | 0.52 |
| **Latency** | ~45ms/req |

## 🏃‍♂️ How to Run

### 1. Using Docker (Recommended)
```bash
# Build the image
docker build -t fraud-api ./app

# Run the container
docker run -p 8000:8000 fraud-api