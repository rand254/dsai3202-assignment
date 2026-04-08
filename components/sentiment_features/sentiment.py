import argparse
import os
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    # Needed for fresh Azure ML compute
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    # ✅ Read from Azure folder
    file_path = os.path.join(args.data, "data.parquet")
    df = pd.read_parquet(file_path)

    # ✅ Handle nulls safely
    df["reviewText"] = df["reviewText"].fillna("").astype(str)

    # ✅ Compute sentiment scores
    scores = df["reviewText"].apply(sia.polarity_scores)

    df["sentiment_pos"] = scores.apply(lambda s: s["pos"])
    df["sentiment_neg"] = scores.apply(lambda s: s["neg"])
    df["sentiment_neu"] = scores.apply(lambda s: s["neu"])
    df["sentiment_compound"] = scores.apply(lambda s: s["compound"])

    # ✅ Save output correctly
    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)

    print("Sentiment features done.")
    print("Rows:", len(df))


if __name__ == "__main__":
    main()