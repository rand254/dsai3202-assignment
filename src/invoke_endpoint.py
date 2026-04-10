import requests
import json
import pandas as pd
from sklearn.metrics import accuracy_score

# -------------------------------
# Endpoint details
# -------------------------------
ENDPOINT_URL = ""
API_KEY = ""

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# -------------------------------
# Load data
# -------------------------------
def load_data(path):
    return pd.read_parquet(path)

# -------------------------------
# Create labels (same as training)
# -------------------------------
def create_labels(df):
    df["label"] = (df["overall"] >= 4).astype(int)
    return df

# -------------------------------
# Build features (EXACT SAME LOGIC)
# -------------------------------
def build_features(df):
    X = df.drop(columns=["overall", "label"], errors="ignore")
    X = X.select_dtypes(include=["number"])
    X = X.fillna(0)
    return X

# -------------------------------
# Main
# -------------------------------
def main():

    # 🔥 CHANGE THIS PATH
    deploy_path = "data/data.parquet"

    print("Loading deployment data...")
    df = load_data(deploy_path)

    print("Creating labels...")
    df = create_labels(df)

    print("Building features...")
    X = build_features(df)
    y_true = df["label"]

    print("Sending request to endpoint...")

    payload = {
        "data": X.values.tolist()
    }

    response = requests.post(
        ENDPOINT_URL,
        headers=headers,
        data=json.dumps(payload)
    )

    result = response.json()

    print("Raw response:", result)

    # -------------------------------
    # Extract predictions
    # -------------------------------
    preds = result["predictions"]

    # -------------------------------
    # Evaluate
    # -------------------------------
    acc = accuracy_score(y_true, preds)

    print("\n FINAL RESULTS:")
    print("Predictions (first 10):", preds[:10])
    print(f"Deployment Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()