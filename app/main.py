import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

# Initialize the App
app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="An XGBoost-based API to detect fraudulent transactions.",
    version="1.0"
)

# Global variables to hold our artifacts
model = None
encoder = None
threshold = 0.5
freq_maps = {}
model_columns = []

# This runs ONCE when the server starts
@app.on_event("startup")
def load_artifacts():
    global model, encoder, threshold, freq_maps, model_columns
    
    print("Loading artifacts...")
    
    # 1. Load Model
    model = xgb.Booster()
    model.load_model("xgb_fraud_model.json")
    
    # 2. Load Ordinal Encoder
    encoder = joblib.load("ordinal_encoder.joblib")
    
    # 3. Load Threshold
    with open("threshold.json", "r") as f:
        threshold = json.load(f)["threshold"]
        
    # 4. Load Frequency Maps
    with open("frequency_maps.json", "r") as f:
        freq_maps = json.load(f)

    # 5. Load Expected Column Order
    with open("model_columns.json", "r") as f:
        model_columns = json.load(f)["columns"]
        
    print(f"Artifacts loaded. Model ready with threshold: {threshold}")

# ----- 2. The Input Data Model -------
# Use Pydantic to define what a valid transaction looks like.
# This automatically rejects bad data (e.g., if someone sends a string for TransactionAmt).
class Transaction(BaseModel):
    # We list the raw fields we expect from the user
    TransactionID: int
    TransactionDT: int
    TransactionAmt: float
    ProductCD: str
    card1: int
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    # Add other columns as necessary based on your training data...
    
    # Use 'dict' to allow extra fields without crashing, 
    # or define every single field used in training.
    class Config:
        extra = "allow"

# ------- 3: The Preprocessing Logic
# This function takes the raw JSON and transforms it into the exact DataFrame format the model was trained on.
# It replicates the Feature Engineering we did in the notebook (Time, Frequency, Encoding).
def preprocess_input(data: dict) -> pd.DataFrame:
    # 1. Convert dictionary to DataFrame (1 row)
    df = pd.DataFrame([data])
    
    # 2. Feature Engineering: Time
    df['hour'] = (df['TransactionDT'] // 3600) % 24
    df['day'] = (df['TransactionDT'] / (3600 * 24)) // 1
    
    # 3. Feature Engineering: Frequency Encoding
    # Optimization: Build a dictionary first, then add all at once
    freq_data = {}
    for col, mapping in freq_maps.items():
        val = df.iloc[0].get(col)
        # Handle type mismatches (JSON keys are strings)
        freq = mapping.get(str(val)) 
        if freq is None:
            freq = mapping.get(val, 0.0)
        freq_data[col + '_freq'] = freq
        
    # Add frequency columns to df
    if freq_data:
        df = pd.concat([df, pd.DataFrame([freq_data])], axis=1)

    # 4. Handle Missing Columns (Optimized for Performance)
    # Identify which columns the model needs that we don't have yet
    missing_cols = [col for col in model_columns if col not in df.columns]
    
    if missing_cols:
        # Create a DataFrame of missing columns filled with -999
        # We use pd.concat instead of a loop to avoid "DataFrame Fragmentation" warning
        df_missing = pd.DataFrame(-999, index=df.index, columns=missing_cols)
        df = pd.concat([df, df_missing], axis=1)

    # 5. Apply Ordinal Encoding (Strings -> Numbers)
    cat_cols_encoded = getattr(encoder, "feature_names_in_", [])
    cols_to_encode = [c for c in cat_cols_encoded if c in df.columns]
    
    if cols_to_encode:
        # A. Fill missing values with "Unknown"
        df[cols_to_encode] = df[cols_to_encode].fillna("Unknown")
        
        # B. CRITICAL FIX: Force columns to String type
        # This prevents the "ufunc isnan" crash on words like "visa"
        df[cols_to_encode] = df[cols_to_encode].astype(str)
        
        # C. Transform and cast to float
        df[cols_to_encode] = encoder.transform(df[cols_to_encode]).astype(float)

    # 6. Reorder columns to match EXACTLY what the model expects
    df = df[model_columns]
    
    # 7. Convert to DMatrix (XGBoost format)
    dmatrix = xgb.DMatrix(df)
    
    return dmatrix

# ------- 4: The Prediction Endpoint -------
@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # 1. Convert Pydantic object to dict
        data_dict = transaction.dict()
        
        # 2. Preprocess
        processed_data = preprocess_input(data_dict)
        
        # 3. Predict Probability
        # XGBoost predict returns a numpy array of probabilities
        prob = model.predict(processed_data)[0]
        
        # 4. Apply Threshold
        is_fraud = bool(prob > threshold)
        
        # 5. Return JSON
        return {
            "transaction_id": transaction.TransactionID,
            "fraud_probability": float(prob),
            "is_fraud": is_fraud,
            "status": "BLOCKED" if is_fraud else "APPROVED"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ------- 5: Health Check Endpoint -------
