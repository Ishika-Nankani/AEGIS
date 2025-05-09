import pandas as pd
import yaml
import argparse
import os

def load_config(config_path):
    """Load and parse the YAML configuration file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def generalize(values, M):
    """Convert numeric values into M buckets, represented as ranges"""
    values = pd.Series(values).astype(float)
    min_val = values.min()
    max_val = values.max()
    
    if min_val == max_val:
        return pd.Series([f"{int(min_val)}:{int(min_val)}" for _ in values.index], index=values.index)
    
    total_range = max_val - min_val
    bucket_size = total_range // M
    remainder = int(total_range % M)
    
    buckets = []
    start = int(min_val)
    for i in range(M):
        extra = 1 if i < remainder else 0
        end = start + bucket_size + extra
        buckets.append((start, end))
        start = end + 1
    
    def assign_bucket(v):
        for (s, e) in buckets:
            if s <= v <= e:
                return f"{s}:{e}"
        s, e = buckets[-1]
        return f"{s}:{e}"
    
    return values.apply(assign_bucket)

def erase_digits(values, num_digits):
    """Replace the last num_digits digits of numeric values with zeros"""
    def erase(v):
        if pd.isnull(v):
            return v
        v_str = str(int(float(v)))
        if len(v_str) <= num_digits:
            return '0' * len(v_str)
        else:
            prefix = v_str[:-num_digits]
            suffix = '0' * num_digits
            return prefix + suffix
    return values.apply(erase)

def suppress(values):
    """Replace all values with asterisks"""
    return pd.Series(['*' for _ in values.index], index=values.index)

def apply_masking(masked_data, masking_attributes):
    """Apply all masking functions specified in the configuration"""
    for attribute, function_list in masking_attributes.items():
        if attribute not in masked_data.columns:
            continue
        original_values = masked_data[attribute]
        if not isinstance(function_list, list) or not function_list:
            continue
        
        for func_info in function_list:
            func_name = func_info.get('function')
            params = func_info.get('params', {})
            
            if func_name == 'generalize':
                M = params.get('M')
                if M is None:
                    continue
                original_values = generalize(original_values, M)
            
            elif func_name == 'erase_digits':
                num_digits = params.get('num_digits', 1)
                original_values = erase_digits(original_values, num_digits)
            
            elif func_name == 'suppress':
                original_values = suppress(original_values)
            
            elif func_name is not None:
                print(f"Warning: Unrecognized masking function '{func_name}' for attribute '{attribute}'")
        
        masked_data[attribute] = original_values
    
    return masked_data

def save_masked_data(masked_data, masked_path):
    """Save the masked data to the specified output path"""
    output_dir = os.path.dirname(masked_path)
    os.makedirs(output_dir, exist_ok=True)
    masked_data.to_csv(masked_path, index=False)

def process_masking(config_path, masked_path):
    """Process the data by applying masking and saving the result"""
    config = load_config(config_path)
    original_path = config['dataset']['original_path']
    masking_attributes = config['masking']['attributes']
    
    original_data = pd.read_csv(original_path)
    masked_data = original_data.copy()
    
    masked_data = apply_masking(masked_data, masking_attributes)
    save_masked_data(masked_data, masked_path)

def main(config_path, masked_path):
    """Main function to run the masking process"""
    process_masking(config_path, masked_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply data masking according to configuration')
    parser.add_argument('--masking_configuration', type=str, required=True, help='Path to the masking configuration YAML file')
    parser.add_argument('--masked_dataset', type=str, required=True, help='Path where the masked dataset will be saved')
    args = parser.parse_args()
    main(args.masking_configuration, args.masked_dataset)
