import argparse
import os
import random
import yaml
import pandas as pd

def parse_args():
    """Parse command line arguments for config generation."""
    parser = argparse.ArgumentParser(description='Generate YAML config files for data masking.')
    parser.add_argument('--original_dataset', type=str, required=True, help='Path to the original CSV data.')
    parser.add_argument('--class_label', type=str, required=True, help='Name of the target variable (excluded from masking).')
    parser.add_argument('--total_configurations', type=int, default=3, help='Number of config files to generate.')
    parser.add_argument('--masking_configurations', type=str, default='configs', help='Directory to save the generated config files.')
    return parser.parse_args()

def random_masking_function(is_numeric=True):
    """Generate a random masking function and parameters based on data type."""
    if is_numeric:
        choices = [
            ('generalize', {'M': random.randint(2, 5)}),
            ('erase_digits', {'num_digits': random.randint(1, 3)}),
            ('suppress', {})
        ]
    else:
        choices = [
            ('suppress', {}),
            (None, None)
        ]
    func, params = random.choice(choices)
    return func, params

def main():
    """Main function to generate masking configuration files."""
    args = parse_args()
    os.makedirs(args.masking_configurations, exist_ok=True)
    
    df = pd.read_csv(args.original_dataset, nrows=200)
    all_columns = df.columns.tolist()
    
    if args.class_label not in all_columns:
        print(f"WARNING: Target variable '{args.class_label}' not found in the CSV columns.")
    
    columns_to_mask = [col for col in all_columns if col != args.class_label]
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c in columns_to_mask]
    categorical_cols = [c for c in columns_to_mask if c not in numeric_cols]
    
    num_digits = len(str(args.total_configurations))
    for i in range(1, args.total_configurations + 1):
        masking_dict = {}
        for col in numeric_cols:
            func, params = random_masking_function(is_numeric=True)
            if func is None:
                continue
            masking_dict.setdefault(col, [])
            masking_dict[col].append({
                'function': func,
                'params': params
            })
        
        for col in categorical_cols:
            func, params = random_masking_function(is_numeric=False)
            if func is None:
                continue
            masking_dict.setdefault(col, [])
            masking_dict[col].append({
                'function': func,
                'params': params
            })
        
        config_name = f"config{i:0{num_digits}d}"
        
        config_data = {
            'dataset': {
                'original_path': args.original_dataset,
                'target_variable': args.class_label
            },
            'masking': {
                'attributes': masking_dict
            }
        }
        
        output_file = os.path.join(args.masking_configurations, f"{config_name}.yaml")
        with open(output_file, 'w') as f:
            yaml.safe_dump(config_data, f, sort_keys=False)

if __name__ == "__main__":
    main()
