import argparse
import os
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.data)

    # assumes normalized review text is in 'reviewText'
    df["review_length_chars"] = df["reviewText"].astype(str).str.len()
    df["review_length_words"] = df["reviewText"].astype(str).str.split().str.len()

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"))

    print("Wrote rows:", len(df))


if __name__ == "__main__":
    main()