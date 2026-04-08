import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    return parser.parse_args()


def load_data(path):
    return pd.read_parquet(os.path.join(path, "data.parquet"))


def save_data(df, path):
    os.makedirs(path, exist_ok=True)
    df.to_parquet(os.path.join(path, "data.parquet"))


def main():
    args = parse_args()

    # Load datasets
    train_df = load_data(args.train)
    val_df = load_data(args.val)
    test_df = load_data(args.test)

    # Extract text column
    train_text = train_df["reviewText"].fillna("")
    val_text = val_df["reviewText"].fillna("")
    test_text = test_df["reviewText"].fillna("")

    # TF-IDF (fit ONLY on train) 
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words="english",
        ngram_range=(1, 2)
    )

    # Fit on train
    train_tfidf = vectorizer.fit_transform(train_text)

    # Transform val + test
    val_tfidf = vectorizer.transform(val_text)
    test_tfidf = vectorizer.transform(test_text)

    # Convert to DataFrame
    feature_names = vectorizer.get_feature_names_out()

    train_features = pd.DataFrame(train_tfidf.toarray(), columns=feature_names)
    val_features = pd.DataFrame(val_tfidf.toarray(), columns=feature_names)
    test_features = pd.DataFrame(test_tfidf.toarray(), columns=feature_names)

    # Keep keys (VERY IMPORTANT for merge step)
    train_features["asin"] = train_df["asin"].values
    train_features["reviewerID"] = train_df["reviewerID"].values

    val_features["asin"] = val_df["asin"].values
    val_features["reviewerID"] = val_df["reviewerID"].values

    test_features["asin"] = test_df["asin"].values
    test_features["reviewerID"] = test_df["reviewerID"].values

    # Save outputs
    save_data(train_features, args.train_out)
    save_data(val_features, args.val_out)
    save_data(test_features, args.test_out)

    print("TF-IDF features created successfully!")


if __name__ == "__main__":
    main()