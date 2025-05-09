import numpy as np
import pandas as pd
import pickle
import argparse

def calculate_tvd(dist1, dist2):
    """Calculate Total Variation Distance between two distributions."""
    total1 = sum(dist1.values())
    total2 = sum(dist2.values())
    support = set(dist1) | set(dist2)
    abs_diff = 0.0
    for key in support:
        p = dist1.get(key, 0) / total1 if total1 > 0 else 0
        q = dist2.get(key, 0) / total2 if total2 > 0 else 0
        abs_diff += abs(p - q)
    return 0.5 * abs_diff

def load_distribution(path):
    """Load distribution from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description="Calculate TVD PUD between two distributions")
    parser.add_argument("--masked", type=str, required=True, help="Path to masked joint distribution")
    parser.add_argument("--reconstructed", type=str, required=True, help="Path to reconstructed joint distribution")
    
    args = parser.parse_args()
    
    masked_dist = load_distribution(args.masked)
    reconstructed_dist = load_distribution(args.reconstructed)
    
    total_pud = 0.0
    count = 0
    
    for attribute in masked_dist:
        if attribute in reconstructed_dist:
            pud = calculate_tvd(masked_dist[attribute], reconstructed_dist[attribute])
            total_pud += pud
            count += 1
    
    if count > 0:
        avg_pud = total_pud / count
        print(f"PUD for TVD = {avg_pud:.6f}")
    else:
        print("No matching attributes found between distributions.")

if __name__ == "__main__":
    main()