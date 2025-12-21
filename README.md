# 🛡️ FraudShield - Real-Time Fraud Detection Engine

A standard end-to-end MLOps project that detects fraudulent transactions in real-time using XGBoost and FastAPI. Containerized with Docker.

![CI Pipeline](https://github.com/YOUR_GITHUB_USERNAME/FraudShield/actions/workflows/test.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)

## 🚀 Key Features
* **Real-Time Inference:** < 100ms latency using FastAPI.
* **Production-Ready Model:** XGBoost Classifier with 0.92 AUC.
* **Imbalance Handling:** Trained using SMOTE techniques.
* **Containerized:** Fully Dockerized for "run anywhere" deployment.
* **Traffic Simulator:** Includes a script to mimic live payment traffic.

## 🏗️ Architecture
The system consists of three core components:
1.  **Model Training Pipeline:** Cleans raw transaction data, engineers features (SMOTE, frequency encoding), and trains an XGBoost classifier.
2.  **Inference Engine (API):** A FastAPI microservice that validates inputs using Pydantic V2 and serves predictions.
3.  **Deployment:** Dockerized environment for consistent execution across cloud platforms.

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
### Option 1: Run with Docker (Recommended)
Build and run the containerized application in two commands:
```bash
# Build the image
docker build -t fraud-api ./app

# Run the container
docker run -p 8000:8000 fraud-api
```

### Option 2: Run Locally (Recommended)
```bash
# Install dependencies
pip install -r app/requirements.txt

# Start the server
cd app
uvicorn main:app --reload
```

## 🧪 Testing & Validation

### Run Unit Tests
This project uses Pytest for robust error checking.
```bash
PYTHONPATH=. pytest
```

### Simulate Live Traffic
To verify the API is working, run the included traffic simulator. It generates random transactions and sends them to the running API.

```bash
python simulate_traffic.py
```

### Output:
> ✅ [APPROVED] Transaction 84920: $45.00 | Fraud Prob: 0.02 🚨 [BLOCKED] Transaction 19283: $5000.00 | Fraud Prob: 0.98

