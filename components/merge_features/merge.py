import argparse
import os
import pandas as pd
import gc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_data", type=str, required=True)
    parser.add_argument("--length", type=str, required=True)
    parser.add_argument("--sentiment", type=str, required=True)
    parser.add_argument("--tfidf", type=str, required=True)
    parser.add_argument("--embedding", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Base Labels
    print("Loading base labels...")
    merged_df = pd.read_parquet(os.path.join(args.original_data, "data.parquet"), columns=["asin", "reviewerID", "overall"])
    merged_df.drop_duplicates(subset=["asin", "reviewerID"], inplace=True)

    # 2. Merge Function with Duplicate Handling
    def safe_merge(base, path, name, cols=None):
        print(f"Merging {name}...")
        df = pd.read_parquet(os.path.join(path, "data.parquet"))
        
        # Trim columns if specified (for TFIDF/Embeddings)
        if cols:
            actual_cols = ["asin", "reviewerID"] + [c for c in df.columns if c in cols]
            df = df[actual_cols]
        
        if "overall" in df.columns:
            df.drop(columns=["overall"], inplace=True)
            
        # THE FIX: Ensure unique keys before merging
        df.drop_duplicates(subset=["asin", "reviewerID"], inplace=True)
        
        base = base.merge(df, on=["asin", "reviewerID"], how="inner")
        del df
        gc.collect()
        return base

    # 3. Sequential Merges
    merged_df = safe_merge(merged_df, args.length, "length")
    merged_df = safe_merge(merged_df, args.sentiment, "sentiment")
    
    # Reduced TF-IDF (100)
    tfidf_cols = [f"tfidf_{i}" for i in range(100)] 
    merged_df = safe_merge(merged_df, args.tfidf, "tfidf", cols=tfidf_cols)
    
    # Reduced Embeddings (128)
    embed_cols = [f"embedding_{i}" for i in range(128)]
    merged_df = safe_merge(merged_df, args.embedding, "embeddings", cols=embed_cols)

    # 4. Save
    print("Saving final dataset...")
    os.makedirs(args.out, exist_ok=True)
    merged_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print(f"Final shape: {merged_df.shape}")

if __name__ == "__main__":
    main()