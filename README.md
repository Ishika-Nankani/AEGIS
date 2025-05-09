# AEGIS - A Correlation-Based Data Masking Advisor for Data-Sharing Ecosystems

## Masking Configuration Generation

To generate masking configurations for a dataset, use:

```bash
make generate_masking_configurations ORIGINAL_DATA="path/to/your/dataset.csv" CLASS_LABEL="target_column_name" TOTAL_CONFIGURATIONS=10 CONFIG_DIR="configs"
```

### Parameters

- `ORIGINAL_DATA` - Path to the original CSV dataset
- `CLASS_LABEL` - Name of the target variable (excluded from masking)
- `TOTAL_CONFIGURATIONS` - Number of masking configurations to generate
- `CONFIG_DIR` - Directory where configurations will be stored (default: configs)

### Output

The command will generate YAML configuration files in the `configs` directory. Each configuration includes:

1. Dataset path information
2. Masking functions to be applied to each attribute

Example configuration:
```yaml
dataset:
  original_path: path/to/your/dataset.csv
  target_variable: target_column_name
masking:
  attributes:
    age:
    - function: generalize
      params:
        M: 3
    income:
    - function: suppress
      params: {}
```

## Applying Data Masking

To apply a masking configuration and generate a masked dataset, use:

```bash
make mask_data MASKING_CONFIG="configs/config_1.yaml" MASKED_DATA="output/masked_dataset.csv"
```

### Parameters

- `MASKING_CONFIG` - Path to a YAML masking configuration file
- `MASKED_DATA` - Path to the output masked dataset (including directory)

### Output

The command will apply the specified masking configuration to the original dataset (defined in the configuration file) and save the masked data to the specified output location.

## Computing Distributions

Before performing data reconstruction, you need to compute distributions. Use:

```bash
make compute_distribution DATASET="path/to/dataset.csv" MARGINALS="path/to/marginals.pkl" JOINT_DISTRIBUTION="path/to/joint_distribution.pkl" CLASS_LABEL="target_column_name"
```

### Parameters

- `DATASET` - Path to the input dataset CSV file
- `MARGINALS` - Path to save marginals (frequency distributions for each attribute)
- `JOINT_DISTRIBUTION` - Path to save joint distributions with class label
- `CLASS_LABEL` - Name of the target/class variable column

### Output

The command computes:
1. Marginals: frequency counts for each value in each attribute
2. Joint distributions: joint distributions between each attribute and the class label

These distributions are saved as pickle files and are required for the reconstruction algorithms.

## Evaluating Predictive Utility Deviation (PUD)

To evaluate the PUD between masked and reconstructed distributions, use:

```bash
make evaluate_pud METRIC="chi2" MASKED_JOINT_DIST="path/to/masked.pkl" RECONSTRUCTED_JOINT_DIST="path/to/reconstructed.pkl"
```

### Parameters

- `METRIC` - Metric to use for evaluation (options: chi2, mi, tvd, g3)
- `MASKED_JOINT_DIST` - Path to masked joint distribution pickle file
- `RECONSTRUCTED_JOINT_DIST` - Path to reconstructed joint distribution pickle file

### Output

The command calculates and displays PUD across all attributes

Available metrics:
- `chi2` - Chi-square statistic
- `mi` - Mutual Information
- `tvd` - Total Variation Distance
- `g3` - G3 metric

## Evaluating Model Performance

To run a machine learning model on a dataset and evaluate its performance, use:

```bash
make model DATA="path/to/dataset.csv" MODEL="path/to/model_file.py" TARGET="target_column_name"
```

### Parameters

- `DATA` - Path to the dataset CSV file to evaluate
- `MODEL` - Path to the model Python file to use for evaluation
- `TARGET` - Name of the target variable column

### Output

The command will train the specified model on the dataset and output its accuracy.