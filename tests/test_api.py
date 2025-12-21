from fastapi.testclient import TestClient
from app.main import app

def test_health_check():
    """Ensure the API starts up and docs are reachable"""
    # Using 'with' triggers the startup event (loading the model)
    with TestClient(app) as client:
        response = client.get("/docs")
        assert response.status_code == 200

def test_prediction_valid_transaction():
    """Test a standard valid transaction"""
    with TestClient(app) as client:
        payload = {
            "TransactionID": 1234567,
            "TransactionDT": 150000,
            "TransactionAmt": 50.0,
            "ProductCD": "W",
            "card1": 10000,
            "card2": 111.0,
            "card3": 150.0,
            "card4": "visa",
            "card5": 226.0,
            "card6": "debit",
            "addr1": 123.0,
            "addr2": 87.0,
            "P_emaildomain": "gmail.com",
            "R_emaildomain": "gmail.com"
        }
        response = client.post("/predict", json=payload)
        
        # Debugging: Print error if it fails
        if response.status_code != 200:
            print("\n\n⚠️ SERVER CRASH DETAILS:")
            print(response.text)
            print("-" * 30 + "\n")

        # Check success
        assert response.status_code == 200
        
        # Check response structure
        data = response.json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert isinstance(data["is_fraud"], bool)

def test_prediction_missing_field():
    """Test that the API rejects bad data"""
    with TestClient(app) as client:
        bad_payload = {"TransactionID": 1234567}
        response = client.post("/predict", json=bad_payload)
        
        # Should fail with 422 Validation Error
        assert response.status_code == 422