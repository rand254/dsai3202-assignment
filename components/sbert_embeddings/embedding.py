import argparse
import os
import pandas as pd
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def load_data(path):
    return pd.read_parquet(os.path.join(path, "data.parquet"))


def save_data(df, path):
    os.makedirs(path, exist_ok=True)
    df.to_parquet(os.path.join(path, "data.parquet"))


def main():
    args = parse_args()

    # Load dataset
    df = load_data(args.data)

    # Load SBERT model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Prepare text
    texts = df["reviewText"].fillna("").tolist()

    # Generate embeddings
    embeddings = model.encode(texts)
    embeddings = embeddings[:, :100]

    # Convert to DataFrame
    emb_df = pd.DataFrame(embeddings)

    # Add keys for merging (VERY IMPORTANT)
    emb_df["asin"] = df["asin"].values
    emb_df["reviewerID"] = df["reviewerID"].values

    # Save output
    save_data(emb_df, args.out)

    print("SBERT embeddings created successfully!")


if __name__ == "__main__":
    main()