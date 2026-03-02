import argparse
import os
import re
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # replace numbers
    text = re.sub(r"\d+", " ", text)

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # trim whitespace
    text = text.strip()

    return text


def main():
    args = parse_args()

    df = pd.read_parquet(args.data)

    # assume review text column is called 'reviewText'
    df["reviewText"] = df["reviewText"].apply(clean_text)

    # remove very short reviews (<10 characters)
    df = df[df["reviewText"].str.len() >= 10]

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"))

    print("Rows after normalization:", len(df))


if __name__ == "__main__":
    main()