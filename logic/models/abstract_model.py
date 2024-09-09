import abc
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from ta import add_all_ta_features


class AbstractModel(abc.ABC, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.pca = None
        self.model = None

    @abc.abstractmethod
    def build_model(self):
        """Build and return the actual model."""
        pass

    @abc.abstractmethod
    def preprocess_data(self, data, fit=False):
        """Preprocess the data, including feature engineering, scaling, and PCA."""
        pass

    @abc.abstractmethod
    def train(self, train_data, val_data, num_epochs):
        """Train the model using the provided training and validation data."""
        pass

    @abc.abstractmethod
    def evaluate(self, test_data):
        """Evaluate the model's performance on test data."""
        pass

    def save_model(self, path):
        """Save the trained model to a file."""
        if isinstance(self.model, nn.Module):
            torch.save(self.model.state_dict(), path)
        elif isinstance(self.model, BaseEstimator):
            joblib.dump(self.model, path)
        else:
            raise NotImplementedError("Saving not implemented for this model type")

    def load_model(self, path):
        """Load a trained model from a file."""
        if isinstance(self.model, nn.Module):
            self.model.load_state_dict(torch.load(path))
        elif isinstance(self.model, BaseEstimator):
            import joblib
            self.model = joblib.load(path)
        else:
            raise NotImplementedError("Loading not implemented for this model type")

    def get_features_and_target(self, data):
        target = self.config['target']
        data['target'] = data[target].shift(-1)  # Shift the target to predict next minute's close

        # Calculate technical indicators
        data = add_all_ta_features(data, "Open", "High", "Low", "Close", "Volume", fillna=True)

        # Drop the last row as it won't have a target value
        data = data.dropna().reset_index(drop=True)

        look_ahead_indicators = ['trend_ichimoku_a', 'trend_ichimoku_b', 'trend_visual_ichimoku_a',
                                 'trend_visual_ichimoku_b', 'trend_stc', 'trend_psar_up', 'trend_psar_down']

        # Drop OHLCV columns from the dataset, keeping only the indicators and target
        feature_columns = [col for col in data.columns if col not in
                           (['date', 'Open', 'High', 'Low', 'Volume', 'target'] + look_ahead_indicators)]

        # Ensure all feature columns exist in the dataframe
        feature_columns = [col for col in feature_columns if col in data.columns]

        return data[feature_columns].values, data['target'].values


def set_up_folders():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    subfolder = os.path.join(project_root, 'results', 'outputs')
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    return project_root, subfolder

def evaluate_dollar_difference(model, data_loader, scaler_y, device):
    model.eval()
    total_abs_error = 0
    count = 0

    # Check the type of scaler_y
    if not isinstance(scaler_y, StandardScaler):
        raise TypeError(f"Expected StandardScaler, but got {type(scaler_y)}")

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred, _ = model(X_batch)

            # Log shapes for debugging
            logging.debug(f"y_pred shape: {y_pred.shape}, y_batch shape: {y_batch.shape}")

            # Ensure y_pred and y_batch have the correct shape
            y_pred = y_pred.view(-1, 1)
            y_batch = y_batch.view(-1, 1)

            # Convert to numpy and reshape if necessary
            y_pred_np = y_pred.cpu().numpy()
            y_batch_np = y_batch.cpu().numpy()

            try:
                # Convert predictions and targets back to the original scale
                y_pred_unscaled = scaler_y.inverse_transform(y_pred_np)
                y_batch_unscaled = scaler_y.inverse_transform(y_batch_np)

                # Calculate the absolute error
                total_abs_error += np.sum(np.abs(y_pred_unscaled - y_batch_unscaled))
                count += len(y_batch)
            except ValueError as e:
                logging.error(f"Error in inverse transform: {str(e)}")
                logging.error(f"y_pred_np shape: {y_pred_np.shape}, y_batch_np shape: {y_batch_np.shape}")
                raise

    if count == 0:
        raise ValueError("No samples were processed")

    average_dollar_diff = total_abs_error / count
    return average_dollar_diff


def save_experiment_results(training_time, avg_time_per_epoch, test_loss, avg_dollar_diff, data_limit, pca, csv_path):
    results = {
        'Training Time (seconds)': [f"{training_time:.2f}"],
        'Average Time per Epoch (seconds)': [f"{avg_time_per_epoch:.2f}"],
        'Test Loss': [f"{test_loss:.6f}"],
        'Average Dollar Difference ($)': [f"{avg_dollar_diff:.2f}"],
        'Data Limit': [data_limit],
        'PCA': [pca]
    }

    df = pd.DataFrame(results)

    # Check if the CSV file exists to append or create it
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False, float_format='%.6f')
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False, float_format='%.6f')