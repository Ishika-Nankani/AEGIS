import numpy as np
import pandas as pd
import pickle
import argparse

def compute_contingency_from_distribution(distribution):
    """Convert a distribution into a contingency table (DataFrame)."""
    table = {}
    for (x, y), count in distribution.items():
        table.setdefault(x, {})
        table[x][y] = table[x].get(y, 0) + count
    df = pd.DataFrame(table).fillna(0).T
    return df

def compute_chi_square(contingency):
    """Compute the Chi-square statistic for a given contingency table."""
    total = contingency.values.sum()
    row_totals = contingency.sum(axis=1)
    col_totals = contingency.sum(axis=0)
    chi2 = 0.0
    for row in contingency.index:
        for col in contingency.columns:
            expected = (row_totals[row] * col_totals[col]) / total if total > 0 else 0
            observed = contingency.at[row, col]
            if expected > 0:
                chi2 += ((observed - expected) ** 2) / expected
    return chi2

def calculate_chi2(dist1, dist2):
    """Calculate normalized Chi-square difference between two distributions."""
    contingency1 = compute_contingency_from_distribution(dist1)
    contingency2 = compute_contingency_from_distribution(dist2)
    
    chi2_1 = compute_chi_square(contingency1)
    chi2_2 = compute_chi_square(contingency2)
    
    total1 = sum(dist1.values())
    total2 = sum(dist2.values())
    
    norm_chi2_1 = chi2_1 / total1 if total1 > 0 else 0
    norm_chi2_2 = chi2_2 / total2 if total2 > 0 else 0
    
    return abs(norm_chi2_1 - norm_chi2_2)

def load_distribution(path):
    """Load a distribution from a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description="Calculate Chi-square PUD between two distributions")
    parser.add_argument("--masked", type=str, required=True, help="Path to masked joint distribution")
    parser.add_argument("--reconstructed", type=str, required=True, help="Path to reconstructed joint distribution")
    
    args = parser.parse_args()
    
    masked_dist = load_distribution(args.masked)
    reconstructed_dist = load_distribution(args.reconstructed)
    
    total_pud = 0.0
    count = 0
    
    for attribute in masked_dist:
        if attribute in reconstructed_dist:
            pud = calculate_chi2(masked_dist[attribute], reconstructed_dist[attribute])
            total_pud += pud
            count += 1
    
    if count > 0:
        avg_pud = total_pud / count
        print(f"PUD for Chi-square = {avg_pud:.6f}")
    else:
        print("No matching attributes found between distributions.")

if __name__ == "__main__":
    main()
