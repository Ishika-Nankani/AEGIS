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

def aegis_with_1D(row_sums, col_sums, max_iter=1000, tol=1e-6):
    """Implements the IPF algorithm with marginal constraints"""
    matrix = np.ones((len(row_sums), len(col_sums)), dtype=float)
    for _ in range(max_iter):
        rs = matrix.sum(axis=1)
        rs[rs == 0] = 1e-12
        for i in range(len(row_sums)):
            matrix[i, :] *= row_sums[i] / rs[i]
        cs = matrix.sum(axis=0)
        cs[cs == 0] = 1e-12
        for j in range(len(col_sums)):
            matrix[:, j] *= col_sums[j] / cs[j]
        new_rs = matrix.sum(axis=1)
        new_cs = matrix.sum(axis=0)
        if np.allclose(new_rs, row_sums, atol=tol) and np.allclose(new_cs, col_sums, atol=tol):
            break
    return matrix

def reconstruct(masked_joint_distribution_path, original_marginal_path, reconstructed_joint_distribution_path, max_iter=1000, tol=1e-6):
    """Reconstructs the joint distribution using IPF with marginals"""
    masked_joint_distribution = load(masked_joint_distribution_path)
    original_marginal = load(original_marginal_path)
    
    reconstructed_joint_distribution = {}
    
    for attribute, distributions in masked_joint_distribution.items():
        orig_counts = original_marginal.get(attribute, {})
        orig_values = sorted(orig_counts.keys())
        masked_distribution = pd.DataFrame.from_dict(distributions)
        label_categories = sorted(masked_distribution.columns)
        
        reconstructed = {}
        
        for bucket in masked_distribution.index:
            rng = parse_range_bucket(str(bucket))
            if rng is None:
                continue
                
            mb_start, mb_end = rng
            candidates = [v for v in orig_values if mb_start <= float(v) <= mb_end]
            
            if not candidates:
                continue
                
            row_sums = [orig_counts.get(v, 0) for v in candidates]
            col_sums = [masked_distribution.at[bucket, label] if label in masked_distribution.columns else 0 
                        for label in label_categories]
            
            matrix = aegis_with_1D(row_sums, col_sums, max_iter, tol)
            
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
    parser = argparse.ArgumentParser(description="Reconstruct data using AEGIS with 1D marginals")
    parser.add_argument("--masked-joint-distribution", type=str, required=True, 
                        help="Path to masked joint distribution")
    parser.add_argument("--original-marginal", type=str, required=True, 
                        help="Path to original marginal")
    parser.add_argument("--reconstructed-joint-distribution", type=str, required=True, 
                        help="Path to save reconstructed joint distribution")
    parser.add_argument("--max-iter", type=int, default=1000, 
                        help="Maximum number of IPF iterations")
    parser.add_argument("--tol", type=float, default=1e-6, 
                        help="Convergence tolerance")
    
    args = parser.parse_args()
    
    reconstruct(args.masked_joint_distribution, args.original_marginal, args.reconstructed_joint_distribution, args.max_iter, args.tol)

if __name__ == "__main__":
    main()