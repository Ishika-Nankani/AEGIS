import os
import pandas as pd
import numpy as np
import pickle
import argparse

def compute_marginals(df):
    """Computes 1D marginals (frequency distributions) for all columns in the dataset"""
    marginals = {}
    for column in df.columns:
        marginals[column] = df[column].value_counts().to_dict()  # Removed normalize=True to get actual counts
    return marginals

def compute_joint_distribution(df, class_label):
    """Computes joint distributions for each column paired with the class label"""
    marginals = {}
    for column in df.columns:
        if column != class_label:
            joint_counts = df.groupby([column, class_label]).size().unstack(fill_value=0)
            marginals[column] = joint_counts.to_dict()
    return marginals

def save(marginals, output_path):
    """Saves marginal distributions to the specified path using pickle"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(marginals, f)

def load_marginals(input_path):
    """Loads marginal distributions from the specified path"""
    with open(input_path, 'rb') as f:
        return pickle.load(f)

def main(dataset_path, marginals_path, joint_distribution_path, class_label):
    """Main function to compute and save marginals from a dataset"""
    df = pd.read_csv(dataset_path)
    
    marginal = compute_marginals(df)
    joint_distribution = compute_joint_distribution(df, class_label)
    
    save(marginal, marginals_path)
    save(joint_distribution, joint_distribution_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and save marginals from a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the input dataset CSV file")
    parser.add_argument("--marginals", type=str, required=True, help="Path to save marginals")
    parser.add_argument("--joint-distribution", type=str, required=True, help="Path to save joint distribution")
    parser.add_argument("--class-label", type=str, required=True, help="Column name of the class label")
    
    args = parser.parse_args()
    
    main(args.dataset, args.marginals, args.joint_distribution, args.class_label)