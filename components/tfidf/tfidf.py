import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--deploy", type=str, required=True) # Added
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--deploy_out", type=str, required=True) # Added
    return parser.parse_args()

def load_data(path):
    return pd.read_parquet(os.path.join(path, "data.parquet"))

def save_data(df, path):
    os.makedirs(path, exist_ok=True)
    df.to_parquet(os.path.join(path, "data.parquet"), index=False)

def main():
    args = parse_args()

    # Load datasets
    train_df = load_data(args.train)
    val_df = load_data(args.val)
    test_df = load_data(args.test)
    deploy_df = load_data(args.deploy) # Added

    # Extract text column
    train_text = train_df["reviewText"].fillna("")
    val_text = val_df["reviewText"].fillna("")
    test_text = test_df["reviewText"].fillna("")
    deploy_text = deploy_df["reviewText"].fillna("") # Added

    # TF-IDF (FIT ONLY ON TRAIN)
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words="english",
        ngram_range=(1, 2)
    )

    # Fit on train
    train_tfidf = vectorizer.fit_transform(train_text)

    # Transform others
    val_tfidf = vectorizer.transform(val_text)
    test_tfidf = vectorizer.transform(test_text)
    deploy_tfidf = vectorizer.transform(deploy_text) # Added

    # Convert to DataFrame
    feature_names = [f"tfidf_{i}" for i in range(len(vectorizer.get_feature_names_out()))]

    def create_feat_df(tfidf_matrix, original_df, names):
        df_feat = pd.DataFrame(tfidf_matrix.toarray(), columns=names)
        df_feat["asin"] = original_df["asin"].values
        df_feat["reviewerID"] = original_df["reviewerID"].values
        return df_feat

    train_features = create_feat_df(train_tfidf, train_df, feature_names)
    val_features = create_feat_df(val_tfidf, val_df, feature_names)
    test_features = create_feat_df(test_tfidf, test_df, feature_names)
    deploy_features = create_feat_df(deploy_tfidf, deploy_df, feature_names) # Added

    # Save outputs
    save_data(train_features, args.train_out)
    save_data(val_features, args.val_out)
    save_data(test_features, args.test_out)
    save_data(deploy_features, args.deploy_out) # Added

    print("TF-IDF features created successfully for 4 splits!")

if __name__ == "__main__":
    main()