import os
import itertools
import yaml
import subprocess
import pandas as pd
import re

# Define hyperparameter grids
param_grids = {
    'hidden_dim': [64, 128, 256],
    'num_layers': [1, 2, 3],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'dropout': [0.2, 0.3, 0.5]
}

# Create combinations of hyperparameters
param_combinations = list(itertools.product(*param_grids.values()))

# Path to the existing LSTM script
lstm_script = 'lstm.py'

# Path to the base config file
base_config_path = '../config/config.yaml'

# Results storage
results = []


def run_experiment(params):
    # Load base config
    with open(base_config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update config with current hyperparameters
    config.update({
        'hidden_dim': params[0],
        'num_layers': params[1],
        'learning_rate': params[2],
        'dropout': params[3]
    })

    # Save updated config to a temporary YAML file
    temp_config_path = 'temp_config.yaml'
    with open(temp_config_path, 'w') as file:
        yaml.safe_dump(config, file)

    # Run the LSTM training script with the updated config
    try:
        result = subprocess.run(['python', lstm_script, temp_config_path],
                                capture_output=True, text=True, check=True)

        # Extract loss values from the output
        output = result.stdout + result.stderr  # Combine stdout and stderr

        test_loss_match = re.search(r'Test Loss:\s*([\d.]+)', output)
        avg_usd_match = re.search(r'Average Dollar Difference:\s*\$([\d.]+)', output)
        if test_loss_match:
            test_loss = float(test_loss_match.group(1))
            print(f"Test Loss: {test_loss}")
            avg_usd = avg_usd_match.group(1)
            print(f"Average Dollar Difference: {avg_usd}")
            return test_loss
        else:
            print("Test Loss not found in the output.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running script with params {params}: {e}")
        print("Error output:", e.output)
        return None
    finally:
        # Clean up the temporary config file
        os.remove(temp_config_path)


# Perform grid search
for params in param_combinations:
    print(f"Testing parameters: {params}")
    test_loss = run_experiment(params)

    if test_loss is not None:
        results.append({
            'hidden_dim': params[0],
            'num_layers': params[1],
            'learning_rate': params[2],
            'dropout': params[3],
            'test_loss': test_loss
        })

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('hyperparameter_tuning_results.csv', index=False)
print("Hyperparameter tuning results saved to 'hyperparameter_tuning_results.csv'")
