import pandas as pd
import numpy as np
import pickle
import os
import random
import argparse

def load(path):
    """Load distributions from a pickle file"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def parse_range_bucket(bucket_str):
    """Parse a range bucket string (e.g., '10:20') into start and end values"""
    try:
        start, end = bucket_str.split(':')
        return float(start), float(end)
    except Exception:
        return None

def reconstruct(marginal_masked_joint_distribution_path, reconstructed_joint_distribution_path, random_seed=42):
    """Reconstructs the 2D distribution using sampling"""
    random.seed(random_seed)
    np.random.seed(random_seed)

    masked_joint_distribution = load(marginal_masked_joint_distribution_path)
    
    reconstructed_joint_distribution = {}
    
    for attribute, distributions in masked_joint_distribution.items():
        masked_distribution = pd.DataFrame.from_dict(distributions)
        label_categories = sorted(masked_distribution.columns)
        
        reconstructed = {}
        
        for bucket in masked_distribution.index:
            rng = parse_range_bucket(str(bucket))
            if rng is None:
                continue
                
            mb_start, mb_end = rng
            
            label_counts = {}
            for label in label_categories:
                if label in masked_distribution.columns:
                    label_counts[label] = masked_distribution.at[bucket, label]
                else:
                    label_counts[label] = 0
            
            for label, count in label_counts.items():
                if count > 0:
                    sampled_values = np.random.uniform(mb_start, mb_end, int(count))
                    for value in sampled_values:
                        key = (value, label)
                        reconstructed[key] = reconstructed.get(key, 0) + 1
        
        reconstructed_joint_distribution[attribute] = reconstructed
    
    os.makedirs(os.path.dirname(reconstructed_joint_distribution_path), exist_ok=True)
    with open(reconstructed_joint_distribution_path, 'wb') as f:
        pickle.dump(reconstructed_joint_distribution, f)
    
    return reconstructed_joint_distribution

def main():
    parser = argparse.ArgumentParser(description="Reconstruct data using sampling approach")
    parser.add_argument("--masked-joint-distribution", type=str, required=True, 
                        help="Path to masked joint distribution")
    parser.add_argument("--reconstructed-joint-distribution", type=str, required=True, 
                        help="Path to save reconstructed joint distribution")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    reconstruct(args.masked_joint_distribution, args.reconstructed_joint_distribution, args.seed)

if __name__ == "__main__":
    main()