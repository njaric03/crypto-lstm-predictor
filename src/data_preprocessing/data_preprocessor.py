import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from sklearn.decomposition import PCA
import numpy as np


class DataPreprocessor:
    def __init__(self, scaler_type='standard', use_pca=False):
        if scaler_type == 'standard':
            self.scaler_X = StandardScaler()
            self.scaler_volatility = StandardScaler()
            self.scaler_volume = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler_X = MinMaxScaler()
            self.scaler_volatility = MinMaxScaler()
            self.scaler_volume = MinMaxScaler()
        else:
            raise ValueError("Invalid scaler type. Use 'standard' or 'minmax'.")

        self.use_pca = use_pca
        self.pca = None
        self.fitted = False

    def transform_data(self, X, y, volatility, volume):
        X_scaled = self.scaler_X.transform(X)
        y_categorized = self._categorize_y(y)
        volatility_scaled = self.scaler_volatility.transform(volatility.values.reshape(-1, 1)).flatten()
        volume_scaled = self.scaler_volume.transform(volume.values.reshape(-1, 1)).flatten()
        if self.use_pca:
            X_scaled = self.pca.transform(X_scaled)
        return X_scaled, y_categorized, volatility_scaled, volume_scaled

    def _categorize_y(self, y):
        full_window_size = 60
        num_categories = 3
        min_window_size = 5  # Minimum number of values to start categorizing

        categories = np.zeros(len(y), dtype=int)

        for i in range(len(y)):
            window_start = max(0, i - full_window_size + 1)
            window = y[window_start:i + 1]

            if len(window) >= min_window_size:
                rank = np.searchsorted(np.sort(window), y[i])
                percentile = rank / len(window)
                category = min(int(percentile * num_categories), num_categories - 1)
                categories[i] = category
            else:
                categories[i] = 1  # Not enough data to categorize

        return categories

    def fit_transform_data(self, X, y, volatility, volume, subfolder, n_components=0.95):
        if self.fitted:
            return self.transform_data(X, y, volatility, volume)

        X_scaled = self.scaler_X.fit_transform(X)
        torch.save(self.scaler_X, os.path.join(subfolder, 'scaler_X.pth'))

        y_categorized = self._categorize_y(y)

        volatility_scaled = self.scaler_volatility.fit_transform(volatility.values.reshape(-1, 1)).flatten()
        volume_scaled = self.scaler_volume.fit_transform(volume.values.reshape(-1, 1)).flatten()

        if self.use_pca:
            self.pca = PCA(n_components=n_components)
            X_scaled = self.pca.fit_transform(X_scaled)
            torch.save(self.pca, os.path.join(subfolder, "pca.pth"))

        self.fitted = True
        return X_scaled, y_categorized, volatility_scaled, volume_scaled