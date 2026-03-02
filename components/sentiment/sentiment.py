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

    # needed on fresh AzureML compute
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    df = pd.read_parquet(args.data)

    # compound is [-1, +1]
    scores = df["reviewText"].astype(str).apply(sia.polarity_scores)

    df["sentiment_pos"] = scores.apply(lambda s: s["pos"])
    df["sentiment_neg"] = scores.apply(lambda s: s["neg"])
    df["sentiment_neu"] = scores.apply(lambda s: s["neu"])
    df["sentiment_compound"] = scores.apply(lambda s: s["compound"])

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"))

    print("Wrote rows:", len(df))


if __name__ == "__main__":
    main()