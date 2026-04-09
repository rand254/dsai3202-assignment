import argparse
import os
import time
import azureml.mlflow
import mlflow
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
    Automatically selects only numeric columns for training.
    This bypasses the need to know every single column name!
    """
    # 1. Drop the target labels first so they aren't used as features
    to_drop = ["overall", "label"]
    temp_df = df.drop(columns=to_drop, errors='ignore')
    
    # 2. Select ONLY numeric columns (float and int)
    # This automatically ignores reviewerID, reviewText, etc.
    X = temp_df.select_dtypes(include=['number'])
    
    print(f"Features used: {X.columns.tolist()}") # This will print in your Azure logs!
    
    if len(X.columns) == 0:
        raise RuntimeError("Feature matrix is empty. No numeric columns found!")
    
    return X

# --- Evaluation ---
def evaluate(model, X, y, split):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    mlflow.log_metric(f"{split}_accuracy", acc)
    print(f"{split} accuracy:", acc)

def main():
    args = parse_args()
    start_time = time.time()

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

    if len(X_train) == 0:
        raise RuntimeError("Training data is empty. That's concerning.")

    print("Training model...")
    # Using a higher max_iter for convergence with many features
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