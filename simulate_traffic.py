import requests
import random
import time
import uuid

# The URL of your Dockerized API
API_URL = "http://127.0.0.1:8000/predict"

# Sample data pools to generate random transactions
card_types = ["visa", "mastercard", "discover", "american express"]
products = ["W", "C", "H", "R", "S"]
domains = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com"]

def generate_transaction():
    """Generates a random fake transaction"""
    return {
        "TransactionID": random.randint(1000000, 9999999),
        "TransactionDT": random.randint(1000000, 20000000),
        "TransactionAmt": round(random.uniform(5.0, 1000.0), 2),
        "ProductCD": random.choice(products),
        "card1": random.randint(1000, 18000),
        "card2": float(random.randint(100, 600)),
        "card3": 150.0,
        "card4": random.choice(card_types),
        "card5": 226.0,
        "card6": "debit",
        "addr1": float(random.randint(100, 500)),
        "addr2": 87.0,
        "P_emaildomain": random.choice(domains),
        "R_emaildomain": random.choice(domains)
    }

def simulate():
    print(f"🚀 Starting Traffic Simulation to {API_URL}...")
    print("Press Ctrl+C to stop.\n")
    
    transaction_count = 0
    fraud_count = 0
    
    try:
        while True:
            # 1. Generate Data
            data = generate_transaction()
            
            # 2. Send Request
            try:
                response = requests.post(API_URL, json=data)
                result = response.json()
                
                # 3. Print Result
                status = result['status']
                prob = result['fraud_probability']
                
                # Color code the output (Green for OK, Red for Fraud)
                if result['is_fraud']:
                    fraud_count += 1
                    print(f"🚨 [BLOCKED] ID: {result['transaction_id']} | Prob: {prob:.4f}")
                else:
                    print(f"✅ [APPROVED] ID: {result['transaction_id']} | Prob: {prob:.4f}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")

            transaction_count += 1
            
            # Sleep for a random time (0.1 to 1.0 seconds) to mimic real traffic
            time.sleep(random.uniform(0.1, 1.0))
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Simulation Stopped.")
        print(f"Total Transactions: {transaction_count}")
        print(f"Frauds Detected: {fraud_count}")

if __name__ == "__main__":
    simulate()