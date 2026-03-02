import argparse
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, required=True)
    p.add_argument("--val", type=str, required=True)
    p.add_argument("--test", type=str, required=True)

    p.add_argument("--train_out", type=str, required=True)
    p.add_argument("--val_out", type=str, required=True)
    p.add_argument("--test_out", type=str, required=True)

    p.add_argument("--text_col", type=str, default="reviewText")
    p.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    return p.parse_args()

def load_split(folder_path: str) -> pd.DataFrame:
    # Azure ML uri_folder will contain the file(s); in our lab we write data.parquet
    file_path = os.path.join(folder_path, "data.parquet")
    return pd.read_parquet(file_path)

def save_split(df: pd.DataFrame, out_folder: str):
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, "data.parquet")
    df.to_parquet(out_path, index=False)

def add_embeddings(df: pd.DataFrame, model: SentenceTransformer, text_col: str) -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available columns: {list(df.columns)}")

    texts = df[text_col].fillna("").astype(str).tolist()

    # Returns a list/array of vectors
    vectors = model.encode(texts, show_progress_bar=False)

    # Put vector into columns: emb_0, emb_1, ...
    emb_df = pd.DataFrame(vectors)
    emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]

    # Keep original + embeddings
    df_out = df.reset_index(drop=True).copy()
    df_out = pd.concat([df_out, emb_df], axis=1)
    return df_out

def main():
    args = parse_args()

    model = SentenceTransformer(args.model_name)

    train_df = load_split(args.train)
    val_df = load_split(args.val)
    test_df = load_split(args.test)

    train_df = add_embeddings(train_df, model, args.text_col)
    val_df = add_embeddings(val_df, model, args.text_col)
    test_df = add_embeddings(test_df, model, args.text_col)

    save_split(train_df, args.train_out)
    save_split(val_df, args.val_out)
    save_split(test_df, args.test_out)

    print("Embedding done.")
    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)

if __name__ == "__main__":
    main()