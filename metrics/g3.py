import numpy as np
import pandas as pd
import pickle
import argparse

def compute_g3(distribution):
    """Calculate G3 metric for a distribution."""
    x_groups = {}
    for (x, y), cnt in distribution.items():
        x_groups.setdefault(x, {})
        x_groups[x][y] = x_groups[x].get(y, 0) + cnt
    total = sum(distribution.values())
    sum_majority = sum(max(group.values()) for group in x_groups.values() if group)
    G3 = total - sum_majority
    norm = G3 / total if total > 0 else 0
    return G3, norm

def calculate_g3_delta(dist1, dist2):
    """Calculate G3 delta between two distributions."""
    _, norm_g3_1 = compute_g3(dist1)
    _, norm_g3_2 = compute_g3(dist2)
    return abs(norm_g3_1 - norm_g3_2)

def load_distribution(path):
    """Load distribution from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description="Calculate G3 PUD between two distributions")
    parser.add_argument("--masked", type=str, required=True, help="Path to masked joint distribution")
    parser.add_argument("--reconstructed", type=str, required=True, help="Path to reconstructed joint distribution")
    
    args = parser.parse_args()
    
    masked_dist = load_distribution(args.masked)
    reconstructed_dist = load_distribution(args.reconstructed)
    
    total_pud = 0.0
    count = 0
    
    for attribute in masked_dist:
        if attribute in reconstructed_dist:
            pud = calculate_g3_delta(masked_dist[attribute], reconstructed_dist[attribute])
            total_pud += pud
            count += 1
    
    if count > 0:
        avg_pud = total_pud / count
        print(f"PUD for G3 = {avg_pud:.6f}")
    else:
        print("No matching attributes found between distributions.")

if __name__ == "__main__":
    main()
