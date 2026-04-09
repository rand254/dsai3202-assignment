import argparse
import os
import time
import azureml.mlflow
import mlflow
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Arguments ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()

# --- Load data ---
def load_data(path):
    # Azure passes the folder path; we need to point to the parquet file inside
    data_path = os.path.join(path, "data.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Path does not exist: {data_path}")
    return pd.read_parquet(data_path)

# --- Labels ---
def create_labels(df):
    if "overall" not in df.columns:
        raise RuntimeError("Column 'overall' is missing. You had one job.")
    
    # Binary classification: 4-5 stars = 1 (Positive), 1-3 stars = 0 (Negative)
    df["label"] = (df["overall"] >= 4).astype(int)
    return df

# --- Features ---
def build_features(df):
    """
    Automatically selects only numeric columns and handles missing values.
    """
    # 1. Drop the target labels
    to_drop = ["overall", "label"]
    temp_df = df.drop(columns=to_drop, errors='ignore')
    
    # 2. Select ONLY numeric columns
    X = temp_df.select_dtypes(include=['number'])
    
    # --- ADD THIS LINE TO FIX THE ERROR ---
    X = X.fillna(0) 
    # --------------------------------------

    print(f"Features used: {X.columns.tolist()}")
    
    if len(X.columns) == 0:
        raise RuntimeError("Feature matrix is empty. No numeric columns found!")
    
    return X

# --- Evaluation ---
# Add these to your imports at the top
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Updated Evaluation Function ---
def evaluate(model, X, y, split):
    """
    Calculates and logs all required metrics to MLflow.
    """
    preds = model.predict(X)
    # Most models need probability for AUC
    probs = model.predict_proba(X)[:, 1] 

    # Calculate metrics
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, average='binary')
    rec = recall_score(y, preds, average='binary')
    f1 = f1_score(y, preds, average='binary')
    auc = roc_auc_score(y, probs)

    # Log to MLflow with names matching the instructions
    mlflow.log_metric(f"{split}_accuracy", acc)
    mlflow.log_metric(f"{split}_precision", prec)
    mlflow.log_metric(f"{split}_recall", rec)
    mlflow.log_metric(f"{split}_f1_score", f1)
    mlflow.log_metric(f"{split}_AUC", auc)

    print(f"--- {split.upper()} METRICS ---")
    print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")

# --- In your main() function, ensure you call it like this ---
# print("Evaluating...")
# evaluate(model, X_train, y_train, "train")
# evaluate(model, X_val, y_val, "val")
# evaluate(model, X_test, y_test, "test")
def main():
    args = parse_args()
    start_time = time.time()

    # Everything must stay inside this "with" block to be logged!
    with mlflow.start_run():
        print("Loading data...")
        train_df = load_data(args.train_data)
        val_df = load_data(args.val_data)
        test_df = load_data(args.test_data)

        print("Creating labels...")
        train_df = create_labels(train_df)
        val_df = create_labels(val_df)
        test_df = create_labels(test_df)

        print("Building features...")
        X_train = build_features(train_df)
        y_train = train_df["label"]
        X_val = build_features(val_df)
        y_val = val_df["label"]
        X_test = build_features(test_df)
        y_test = test_df["label"]

        print("Training model...")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        print("Evaluating...")
        evaluate(model, X_train, y_train, "train")
        evaluate(model, X_val, y_val, "val")
        evaluate(model, X_test, y_test, "test")

        print("Saving model...")
        os.makedirs(args.output, exist_ok=True)
        model_path = os.path.join(args.output, "model.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        runtime = time.time() - start_time
        mlflow.log_metric("training_runtime_seconds", runtime)
        print("Done.")

if __name__ == "__main__":
    main()