import numpy as np
import pandas as pd
import pickle
import argparse

def compute_contingency_from_distribution(distribution):
    """Compute contingency table from a distribution."""
    table = {}
    for (x, y), count in distribution.items():
        table.setdefault(x, {})
        table[x][y] = table[x].get(y, 0) + count
    df = pd.DataFrame(table).fillna(0).T
    return df

def compute_mutual_information(p_x, p_y, p_xy, epsilon=1e-12):
    """Compute mutual information from probability distributions."""
    mi = 0.0
    p_x = p_x / p_x.sum() if p_x.sum() != 0 else p_x
    p_y = p_y / p_y.sum() if p_y.sum() != 0 else p_y
    total = p_xy.values.sum()
    p_xy = p_xy / total if total != 0 else p_xy
    for x in p_xy.index:
        for y in p_xy.columns:
            p_xy_val = p_xy.at[x, y]
            if p_xy_val > 0:
                mi += p_xy_val * np.log2(p_xy_val / (p_x.get(x, 0) * p_y.get(y, 0) + epsilon))
    return mi

def calculate_mutual_information(dist1, dist2, epsilon=1e-12):
    """Calculate mutual information delta between two distributions."""
    contingency1 = compute_contingency_from_distribution(dist1) 
    contingency2 = compute_contingency_from_distribution(dist2)
    
    total1 = contingency1.values.sum() if not contingency1.empty else 0
    total2 = contingency2.values.sum() if not contingency2.empty else 0
    
    mi1 = compute_mutual_information(contingency1.sum(axis=1), contingency1.sum(axis=0), contingency1, epsilon) if total1 > 0 else 0
    mi2 = compute_mutual_information(contingency2.sum(axis=1), contingency2.sum(axis=0), contingency2, epsilon) if total2 > 0 else 0
    
    return abs(mi1 - mi2)

def load_distribution(path):
    """Load distribution from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description="Calculate Mutual Information PUD between two distributions")
    parser.add_argument("--masked", type=str, required=True, help="Path to masked joint distribution")
    parser.add_argument("--reconstructed", type=str, required=True, help="Path to reconstructed joint distribution")
    
    args = parser.parse_args()
    
    masked_dist = load_distribution(args.masked)
    reconstructed_dist = load_distribution(args.reconstructed)
    
    total_pud = 0.0
    count = 0
    
    for attribute in masked_dist:
        if attribute in reconstructed_dist:
            pud = calculate_mutual_information(masked_dist[attribute], reconstructed_dist[attribute])
            total_pud += pud
            count += 1
    
    if count > 0:
        avg_pud = total_pud / count
        print(f"PUD for MI = {avg_pud:.6f}")
    else:
        print("No matching attributes found between distributions.")

if __name__ == "__main__":
    main()