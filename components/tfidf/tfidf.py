import argparse
import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args():
    p = argparse.ArgumentParser()
    # Inputs are uri_folder paths (folders). Each folder contains data.parquet
    p.add_argument("--train", type=str, required=True)
    p.add_argument("--val", type=str, required=True)
    p.add_argument("--test", type=str, required=True)

    # Outputs are uri_folder paths (folders)
    p.add_argument("--train_out", type=str, required=True)
    p.add_argument("--val_out", type=str, required=True)
    p.add_argument("--test_out", type=str, required=True)

    # Column + settings
    p.add_argument("--text_col", type=str, default="reviewText")
    p.add_argument("--max_features", type=int, default=2000)
    p.add_argument("--ngram_min", type=int, default=1)
    p.add_argument("--ngram_max", type=int, default=2)

    return p.parse_args()


def _read_split(folder_path: str) -> pd.DataFrame:
    # Your split.py writes "data.parquet" into the output folder
    file_path = os.path.join(folder_path, "data.parquet")
    return pd.read_parquet(file_path)


def _ensure_text(series: pd.Series) -> pd.Series:
    # safety: handle nulls and force string
    s = series.fillna("").astype(str)
    return s


def _to_feature_df(X, feature_names, base_df: pd.DataFrame) -> pd.DataFrame:
    # Convert sparse matrix to dense (works fine for max_features like 2000).
    # Keep entity keys for later merge (asin, reviewerID)
    X_dense = X.toarray()
    feat_cols = [f"tfidf_{name}" for name in feature_names]
    feat_df = pd.DataFrame(X_dense, columns=feat_cols)

    out = pd.concat(
        [base_df[["asin", "reviewerID"]].reset_index(drop=True), feat_df],
        axis=1
    )
    return out


def _write_out(df: pd.DataFrame, out_folder: str):
    os.makedirs(out_folder, exist_ok=True)
    df.to_parquet(os.path.join(out_folder, "data.parquet"), index=False)


def main():
    args = parse_args()

    # Read splits (folders that contain data.parquet)
    train_df = _read_split(args.train)
    val_df = _read_split(args.val)
    test_df = _read_split(args.test)

    # Basic checks for required columns
    for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for col in ["asin", "reviewerID", args.text_col]:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {df_name} split.")

    train_text = _ensure_text(train_df[args.text_col])
    val_text = _ensure_text(val_df[args.text_col])
    test_text = _ensure_text(test_df[args.text_col])

    # TF-IDF settings per lab recommendations
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        stop_words="english",
        ngram_range=(args.ngram_min, args.ngram_max),
    )

    # IMPORTANT (lab rule): FIT only on train, then TRANSFORM val/test
    X_train = vectorizer.fit_transform(train_text)
    X_val = vectorizer.transform(val_text)
    X_test = vectorizer.transform(test_text)

    feature_names = vectorizer.get_feature_names_out()

    train_out_df = _to_feature_df(X_train, feature_names, train_df)
    val_out_df = _to_feature_df(X_val, feature_names, val_df)
    test_out_df = _to_feature_df(X_test, feature_names, test_df)

    _write_out(train_out_df, args.train_out)
    _write_out(val_out_df, args.val_out)
    _write_out(test_out_df, args.test_out)

    print("TF-IDF done.")
    print("Train rows:", len(train_out_df), "features:", len(feature_names))
    print("Val rows:", len(val_out_df))
    print("Test rows:", len(test_out_df))


if __name__ == "__main__":
    main()