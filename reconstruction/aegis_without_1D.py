import numpy as np
import pandas as pd
import pickle
import os
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

def aegis_without_1d(num_candidates, col_sums, max_iter=1000, tol=1e-6, alpha=0.1):
    """Implements IPF algorithm without using original marginal"""
    matrix = np.full((num_candidates, len(col_sums)), 0.0)
    for j in range(len(col_sums)):
        matrix[:, j] = col_sums[j] / num_candidates
        
    prev_cost = np.inf
    for _ in range(max_iter):
        col_means = matrix.mean(axis=0)
        cost = np.sum(np.abs(matrix - col_means))
        if abs(prev_cost - cost) < tol:
            break
        prev_cost = cost
        matrix = matrix + alpha * (col_means - matrix)
        for j in range(len(col_sums)):
            col_total = matrix[:, j].sum()
            if col_total != 0:
                matrix[:, j] *= col_sums[j] / col_total
    return matrix

def reconstruct(marginal_joint_distribution_path, reconstructed_joint_distribution_path, max_iter=1000, tol=1e-6, alpha=0.1):
    """Reconstructs the 2D distribution using IPF without original marginals"""
    joint_distribution = load(marginal_joint_distribution_path)
    
    reconstructed_joint_distribution = {}
    
    for attribute, distributions in joint_distribution.items():
        masked_distribution = pd.DataFrame.from_dict(distributions)
        label_categories = sorted(masked_distribution.columns)
        
        reconstructed = {}
        
        for bucket in masked_distribution.index:
            rng = parse_range_bucket(str(bucket))
            if rng is None:
                continue
                
            mb_start, mb_end = rng
            num_candidates = max(1, int((mb_end - mb_start) * 10))
            
            col_sums = [masked_distribution.at[bucket, label] if label in masked_distribution.columns else 0 
                        for label in label_categories]
            
            matrix = aegis_without_1d(num_candidates, col_sums, max_iter, tol, alpha)
            
            candidates = np.linspace(mb_start, mb_end, num_candidates)
            
            for i, candidate in enumerate(candidates):
                for j, label in enumerate(label_categories):
                    key = (candidate, label)
                    reconstructed[key] = reconstructed.get(key, 0) + matrix[i, j]
        
        reconstructed_joint_distribution[attribute] = reconstructed
    
    os.makedirs(os.path.dirname(reconstructed_joint_distribution_path), exist_ok=True)
    with open(reconstructed_joint_distribution_path, 'wb') as f:
        pickle.dump(reconstructed_joint_distribution, f)
    
    return reconstructed_joint_distribution

def main():
    parser = argparse.ArgumentParser(description="Reconstruct data using AEGIS without 1D marginals")
    parser.add_argument("--masked-joint-distribution", type=str, required=True, 
                        help="Path to masked joint distribution")
    parser.add_argument("--reconstructed-joint-distribution", type=str, required=True, 
                        help="Path to save reconstructed joint distribution")
    parser.add_argument("--max-iter", type=int, default=1000, 
                        help="Maximum number of IPF iterations")
    parser.add_argument("--tol", type=float, default=1e-6, 
                        help="Convergence tolerance")
    parser.add_argument("--alpha", type=float, default=0.1, 
                        help="Learning rate for the update")
    
    args = parser.parse_args()
    
    reconstruct(args.masked_joint_distribution, args.reconstructed_joint_distribution, args.max_iter, args.tol, args.alpha)

if __name__ == "__main__":
    main()