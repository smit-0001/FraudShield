import os
import json
import joblib
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Optional
from contextlib import asynccontextmanager

# Define global variables
model = None
encoder = None
threshold = 0.5
freq_maps = {}
model_columns = []

# --- 1. THE LIFESPAN MANAGER (Replaces @app.on_event) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs BEFORE the server starts (Load Artifacts)
    global model, encoder, threshold, freq_maps, model_columns
    
    print("Loading artifacts...")
    
    # Get the ABSOLUTE path of the current file (app/main.py)
    # This fixes the "File Not Found" error when running pytest from root
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Load Model
        model = xgb.Booster()
        model.load_model(os.path.join(BASE_DIR, "xgb_fraud_model.json"))
        
        # Load Encoder
        encoder = joblib.load(os.path.join(BASE_DIR, "ordinal_encoder.joblib"))
        
        # Load Threshold
        with open(os.path.join(BASE_DIR, "threshold.json"), "r") as f:
            threshold = json.load(f)["threshold"]
            
        # Load Frequency Maps
        with open(os.path.join(BASE_DIR, "frequency_maps.json"), "r") as f:
            freq_maps = json.load(f)

        # Load Columns
        with open(os.path.join(BASE_DIR, "model_columns.json"), "r") as f:
            model_columns = json.load(f)["columns"]
            
        print(f"✅ Artifacts loaded successfully. Threshold: {threshold}")
        
    except Exception as e:
        print(f"❌ FATAL: Could not load artifacts. Error: {e}")
        # We don't raise here so the app can start and show the error in logs, 
        # but in production, you might want to crash.

    yield  # The application runs here...
    
    # This code runs AFTER the server stops (Cleanup)
    print("Shutting down...")

# --- 2. INITIALIZE APP WITH LIFESPAN ---
app = FastAPI(
    title="FraudShield API",
    version="1.0",
    lifespan=lifespan  # Connect the lifespan manager
)

class Transaction(BaseModel):
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
    
    # class Config:
        # extra = "allow" 
    model_config = ConfigDict(extra="allow")

def preprocess_input(data: dict) -> pd.DataFrame:
    # 1. Convert dictionary to DataFrame
    df = pd.DataFrame([data])
    
    # 2. Feature Engineering: Time
    df['hour'] = (df['TransactionDT'] // 3600) % 24
    df['day'] = (df['TransactionDT'] / (3600 * 24)) // 1
    
    # 3. Feature Engineering: Frequency Encoding
    freq_data = {}
    for col, mapping in freq_maps.items():
        val = df.iloc[0].get(col)
        freq = mapping.get(str(val)) 
        if freq is None:
            freq = mapping.get(val, 0.0)
        freq_data[col + '_freq'] = freq
        
    if freq_data:
        df = pd.concat([df, pd.DataFrame([freq_data])], axis=1)

    # 4. Handle Missing Columns
    missing_cols = [col for col in model_columns if col not in df.columns]
    if missing_cols:
        df_missing = pd.DataFrame(-999, index=df.index, columns=missing_cols)
        df = pd.concat([df, df_missing], axis=1)

    # 5. Ordinal Encoding
    cat_cols_encoded = getattr(encoder, "feature_names_in_", [])
    cols_to_encode = [c for c in cat_cols_encoded if c in df.columns]
    
    if cols_to_encode:
        df[cols_to_encode] = df[cols_to_encode].fillna("Unknown")
        df[cols_to_encode] = df[cols_to_encode].astype(str)
        df[cols_to_encode] = encoder.transform(df[cols_to_encode]).astype(float)

    # 6. Reorder & DMatrix
    df = df[model_columns]
    return xgb.DMatrix(df)

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        # Use model_dump() instead of .dict() (Pydantic V2 fix)
        data_dict = transaction.model_dump()
        processed_data = preprocess_input(data_dict)
        
        prob = model.predict(processed_data)[0]
        is_fraud = bool(prob > threshold)
        
        return {
            "transaction_id": transaction.TransactionID,
            "fraud_probability": float(prob),
            "is_fraud": is_fraud,
            "status": "BLOCKED" if is_fraud else "APPROVED"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)