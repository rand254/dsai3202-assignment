import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=str, required=True)
    parser.add_argument("--sentiment", type=str, required=True)
    parser.add_argument("--tfidf", type=str, required=True)
    parser.add_argument("--embedding", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load each feature dataset
    length_df = pd.read_parquet(os.path.join(args.length, "data.parquet"))
    sentiment_df = pd.read_parquet(os.path.join(args.sentiment, "data.parquet"))
    tfidf_df = pd.read_parquet(os.path.join(args.tfidf, "data.parquet"))
    embedding_df = pd.read_parquet(os.path.join(args.embedding, "data.parquet"))

    # Merge step by step on entity keys
    merged_df = length_df.merge(
        sentiment_df, on=["asin", "reviewerID"], how="inner"
    )

    merged_df = merged_df.merge(
        tfidf_df, on=["asin", "reviewerID"], how="inner"
    )

    merged_df = merged_df.merge(
        embedding_df, on=["asin", "reviewerID"], how="inner"
    )

    # Write final dataset
    os.makedirs(args.out, exist_ok=True)
    merged_df.to_parquet(os.path.join(args.out, "data.parquet"))

    print("Final merged rows:", len(merged_df))
    print("Final columns:", len(merged_df.columns))


if __name__ == "__main__":
    main()