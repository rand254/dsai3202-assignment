import pandas as pd
import os

# Update this path to where your merged parquet file is 
# (Check your 'data/' folder or wherever you saved the output)
file_path = "data/amazon_review_merged_features_train/data.parquet"

if os.path.exists(file_path):
    df = pd.read_parquet(file_path)
    print("--- Data Check ---")
    print(f"Total Columns: {len(df.columns)}")
    
    # This is the "Heads Up" check [cite: 44, 45]
    required_cols = ['overall', 'asin', 'reviewerID']
    for col in required_cols:
        if col in df.columns:
            print(f"✅ Found column: {col}")
        else:
            print(f"❌ MISSING column: {col}")
            
    # Check for feature prefixes
    features = [c for c in df.columns if 'tfidf' in c or 'sbert' in c]
    print(f"✅ Found {len(features)} feature columns.")
else:
    print(f"File not found at {file_path}. Check your path!")