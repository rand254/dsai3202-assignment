import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.6) 
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--deploy_ratio", type=float, default=0.10)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--deploy_out", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    # Azure-safe data loading
    data_path = os.path.join(args.data, "data.parquet")
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_parquet(args.data)
    
    # --- TEMPORAL SPLIT LOGIC (FOR DEPLOYMENT) ---
    # Sort by year to ensure deployment data is the "future"
    df = df.sort_values("review_year", ascending=True)
    
    # Slice the last 10% for Deployment
    deploy_index = int(len(df) * (1 - args.deploy_ratio))
    deploy_df = df.iloc[deploy_index:]
    remaining_df = df.iloc[:deploy_index]
    
    # --- RANDOM SPLIT LOGIC (FOR TRAIN/VAL/TEST) ---
    # We split the remaining 90% using the desired ratios
    # Combined test/val size relative to remaining data is ~0.33
    test_val_combined_ratio = (args.val_ratio + args.test_ratio) / (1 - args.deploy_ratio)
    
    train_df, temp_df = train_test_split(
        remaining_df, 
        test_size=test_val_combined_ratio, 
        random_state=args.seed,
        shuffle=True 
    )
    
    # Split the 30% temp data equally into Val and Test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=args.seed,
        shuffle=True
    )

    # Save all four outputs
    output_map = [
        (args.train_out, train_df), 
        (args.val_out, val_df), 
        (args.test_out, test_df), 
        (args.deploy_out, deploy_df)
    ]

    for path, data in output_map:
        os.makedirs(path, exist_ok=True)
        data.to_parquet(os.path.join(path, "data.parquet"), index=False)

    print(f"Split complete. Deployment: {len(deploy_df)}, Train: {len(train_df)}")

if __name__ == "__main__":
    main()