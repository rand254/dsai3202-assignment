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

    # Azure ML gives a folder → read data.parquet inside it
    file_path = os.path.join(args.data, "data.parquet")
    df = pd.read_parquet(file_path)

    # Safely handle text column
    text = df["reviewText"].fillna("").astype(str)

    # Create features
    df["review_length_chars"] = text.str.len()
    df["review_length_words"] = text.str.split().str.len()

    # Save output
    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)

    print("Length features done.")
    print("Rows:", len(df))


if __name__ == "__main__":
    main()