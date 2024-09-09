import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ta import add_all_ta_features
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import logging
import traceback
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.decomposition import PCA

from logic.models.abstract_model import set_up_folders, choose_n_components, save_experiment_results
from src.data_preprocessing.data_importer import import_data
from src.utils.config_loader import load_config
from src.utils.data_saving_and_displaying import save_and_display_results

import time


class MeanAbsolutePercentageError(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-8  # Small value to avoid division by zero
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100

project_root, subfolder = set_up_folders()
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


class DynamicAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(DynamicAttention, self).__init__()
        self.feature_layer = nn.Linear(2, hidden_dim, bias=False)
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out, volatility, volume):
        # Combine volatility and volume
        features = torch.cat((volatility.unsqueeze(-1), volume.unsqueeze(-1)), dim=-1)
        dynamic_weights = torch.tanh(self.feature_layer(features))
        attention_weights = torch.softmax(self.attention(lstm_out * dynamic_weights).squeeze(-1), dim=1)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * lstm_out, dim=1)
        return context_vector, attention_weights


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = DynamicAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, volatility, volume):  # Added volume parameter
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        context_vector, attention_weights = self.attention(lstm_out, volatility, volume)  # Added volume
        out = self.fc(context_vector)
        return out.view(-1, 1), attention_weights


class CryptoDataset(Dataset):
    def __init__(self, data, volatility, volume, seq_length):  # Added volume parameter
        self.data = torch.FloatTensor(data)
        self.volatility = torch.FloatTensor(volatility)
        self.volume = torch.FloatTensor(volume)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, idx):
        return (self.data[idx:idx + self.seq_length, :-1],
                self.volatility[idx:idx + self.seq_length],
                self.volume[idx:idx + self.seq_length],  # New line
                self.data[idx + self.seq_length - 1, -1])


def calculate_volatility(data, window_size=20):
    data['log_return'] = np.log(data['Close']) - np.log(data['Close'].shift(1))
    data['volatility'] = data['log_return'].rolling(window=window_size).std()
    return data['volatility'].dropna()


def preprocess_data(data: pd.DataFrame, config, scaler_X=None, scaler_y=None, scaler_volatility=None,
                    scaler_volume=None,
                    pca=None, fit=False):
    target = config['target']

    # Calculate the difference in closing price
    data['Close_diff'] = data['Close'].diff()

    # Shift the target to predict the next period's price change
    data['target'] = data['Close_diff'].shift(-1)

    # Remove the first row which will have NaN for Close_diff
    data = data.dropna().reset_index(drop=True)

    data = add_all_ta_features(data, "Open", "High", "Low", "Close", "Volume", fillna=True)
    data = data.dropna().reset_index(drop=True)

    look_ahead_indicators = ['trend_ichimoku_a', 'trend_ichimoku_b', 'trend_visual_ichimoku_a',
                             'trend_visual_ichimoku_b', 'trend_stc', 'trend_psar_up', 'trend_psar_down']

    feature_columns = [col for col in data.columns if col not in
                       (['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'target'] + look_ahead_indicators)]

    logging.info(f"Number of features before PCA: {len(feature_columns)}")

    # Calculate volatility using the new method
    data['volatility'] = calculate_volatility(data, window_size=config.get('volatility_window_size', 20))

    # Drop the close column
    data = data.drop(columns=['Close'])

    # Drop rows with NaN values in volatility
    data = data.dropna().reset_index(drop=True)

    volatility = data['volatility']
    volume = data['Volume']

    X = data[feature_columns].values
    y = data['target'].values

    if not fit:
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        # Save the scaler into a file for later use
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        torch.save(scaler_X, os.path.join(subfolder, 'scaler_X.pth'))

        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        torch.save(scaler_y, os.path.join(subfolder, 'scaler_y.pth'))

        scaler_volatility = StandardScaler()
        volatility_scaled = scaler_volatility.fit_transform(volatility.values.reshape(-1, 1)).flatten()
        scaler_volume = StandardScaler()
        volume_scaled = scaler_volume.fit_transform(volume.values.reshape(-1, 1)).flatten()

        if config.get('use_pca', False):
            logging.info("PCA is enabled. Determining optimal number of components...")
            n_components = choose_n_components(X_scaled,
                                               variance_threshold=config.get('variance_threshold', 0.95))
            pca = PCA(n_components=n_components)
            X_scaled = pca.fit_transform(X_scaled)
            logging.info(f"PCA applied. Number of components: {n_components}")
            logging.info(f"Variance explained by PCA: {sum(pca.explained_variance_ratio_):.4f}")
        else:
            logging.info("PCA is not enabled.")
    else:
        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()
        volatility_scaled = scaler_volatility.transform(volatility.values.reshape(-1, 1)).flatten()
        volume_scaled = scaler_volume.transform(volume.values.reshape(-1, 1)).flatten()

        if pca is not None:
            X_scaled = pca.transform(X_scaled)
            logging.info(f"PCA transform applied. Number of components: {pca.n_components_}")

    logging.info(f"Number of features after preprocessing: {X_scaled.shape[1]}")

    # Ensure no NaN values
    assert not np.isnan(X_scaled).any(), "NaN values found in features"
    assert not np.isnan(y_scaled).any(), "NaN values found in target"
    assert not np.isnan(volatility_scaled).any(), "NaN values found in volatility"

    return (np.hstack((X_scaled, y_scaled.reshape(-1, 1))), volatility_scaled, volume_scaled,
            scaler_X, scaler_y, scaler_volatility, scaler_volume, pca)


def train_model(model: nn.Module, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience=5):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, volatility_batch, volume_batch, y_batch in train_loader:  # Updated this line
            X_batch, volatility_batch, volume_batch, y_batch = X_batch.to(device), volatility_batch.to(
                device), volume_batch.to(device), y_batch.to(device)  # Updated this line
            optimizer.zero_grad()
            y_pred, _ = model(X_batch, volatility_batch, volume_batch)  # Updated this line

            # Ensure no NaN values in model output
            assert not torch.isnan(y_pred).any(), "NaN values found in model output"

            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, volatility_batch, volume_batch, y_batch in val_loader:  # Updated this line
                X_batch, volatility_batch, volume_batch, y_batch = X_batch.to(device), volatility_batch.to(
                    device), volume_batch.to(device), y_batch.to(device)  # Updated this line
                y_pred, _ = model(X_batch, volatility_batch, volume_batch)  # Updated this line

                # Ensure no NaN values in model output
                assert not torch.isnan(y_pred).any(), "NaN values found in model output"

                loss = criterion(y_pred.squeeze(), y_batch)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(subfolder, 'best_lstm_model.pth'))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info("Early stopping triggered")
            break

        # completed_epochs += 1

    # end_time = time.time()
    # duration = end_time - start_time
    # average_time_per_epoch = duration / completed_epochs if completed_epochs > 0 else 0
    # print(f"Training completed in {duration:.2f} seconds")
    # print(f"Average time per epoch: {average_time_per_epoch:.2f} seconds")


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, volatility, volume, targets in data_loader:  # Updated this line
            inputs, volatility, volume, targets = inputs.to(device), volatility.to(device), volume.to(
                device), targets.to(device)  # Updated this line
            outputs, _ = model(inputs, volatility, volume)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate_dollar_difference(model, data_loader, scaler_y, device):
    model.eval()
    total_abs_error = 0
    count = 0

    if not isinstance(scaler_y, StandardScaler):
        raise TypeError(f"Expected StandardScaler, but got {type(scaler_y)}")

    with torch.no_grad():
        for X_batch, volatility_batch, volume_batch, y_batch in data_loader:  # Updated this line
            X_batch, volatility_batch, volume_batch, y_batch = X_batch.to(device), volatility_batch.to(
                device), volume_batch.to(device), y_batch.to(device)  # Updated this line
            y_pred, _ = model(X_batch, volatility_batch, volume_batch)  # Updated this line

            logging.debug(f"y_pred shape: {y_pred.shape}, y_batch shape: {y_batch.shape}")

            y_pred = y_pred.view(-1, 1)
            y_batch = y_batch.view(-1, 1)

            y_pred_np = y_pred.cpu().numpy()
            y_batch_np = y_batch.cpu().numpy()

            try:
                y_pred_unscaled = scaler_y.inverse_transform(y_pred_np)
                y_batch_unscaled = scaler_y.inverse_transform(y_batch_np)

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


def main(config_path):
    # Load configuration
    config = load_config(config_path)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    csv_path = os.path.join(subfolder, 'times.csv')

    try:
        logging.info("Starting main function")
        logging.info(f"Configuration loaded from: {config_path}")

        # Define paths for datasets
        datasets = {
            'train': [os.path.join(project_root, 'data', path) for path in config['train_data']],
            'val': [os.path.join(project_root, 'data', path) for path in config['val_data']],
            'test': [os.path.join(project_root, 'data', path) for path in config['test_data']]
        }

        processed_data = {}
        data_loaders = {}
        scaler_X = None
        scaler_y = None
        scaler_volatility = None
        pca = None

        # Process each dataset (train, val, test)
        for dataset_name, data_path in datasets.items():
            logging.info(f"Processing {dataset_name} dataset from {data_path}")
            data = import_data(data_path, limit=config.get('data_limit', None))
            logging.info(f"Data imported for {dataset_name}, shape: {data.shape}")

            if dataset_name == 'train':
                processed_data[
                    dataset_name], preprocessed_volatility, preprocessed_volume, scaler_X, scaler_y, scaler_volatility, scaler_volume, pca = preprocess_data(
                    data, config, fit=False)
            else:
                processed_data[
                    dataset_name], preprocessed_volatility, preprocessed_volume, _, _, _, _, _ = preprocess_data(data,
                                                                                                                 config,
                                                                                                                 scaler_X,
                                                                                                                 scaler_y,
                                                                                                                 scaler_volatility,
                                                                                                                 scaler_volume,
                                                                                                                 pca,
                                                                                                                 fit=True)

            logging.info(f"Data preprocessed for {dataset_name}, shape: {processed_data[dataset_name].shape}")

            dataset = CryptoDataset(processed_data[dataset_name], preprocessed_volatility, preprocessed_volume,
                                    seq_length=config['seq_length'])
            data_loaders[dataset_name] = DataLoader(dataset, batch_size=config['batch_size'],
                                                    shuffle=(dataset_name == 'train'))
            logging.info(f"DataLoader created for {dataset_name}")
        input_dim = processed_data['train'].shape[1] - 1  # Exclude target column
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
        dropout = config['dropout']
        model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1,
                          dropout=dropout)
        logging.info(f"Model initialized with hidden_dim: {hidden_dim}, num_layers: {num_layers}, dropout: {dropout}")

        # Define the loss function and optimizer
        criterion = MeanAbsolutePercentageError()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        logging.info(f"Loss function, optimizer, and scheduler initialized. Learning rate: {config['learning_rate']}")

        # Train the model
        logging.info("Starting model training")
        start_time = time.time()
        train_model(model, data_loaders['train'], data_loaders['val'], criterion, optimizer, scheduler,
                    config['num_epochs'])
        end_time = time.time()
        training_time = end_time - start_time
        avg_time_per_epoch = training_time / config['num_epochs']
        logging.info("Model training completed")

        # Evaluate the model on the test set
        logging.info("Starting model evaluation on test set")
        model.load_state_dict(torch.load(os.path.join(subfolder, 'best_lstm_model.pth')))
        model.eval()
        test_loss = 0
        test_actuals = []
        test_predictions = []
        with torch.no_grad():
            for X_batch, volatility_batch, volume_batch, y_batch in data_loaders['test']:  # Updated this line
                X_batch, volatility_batch, volume_batch, y_batch = X_batch.to(device), volatility_batch.to(
                    device), volume_batch.to(device), y_batch.to(device)  # Updated this line
                y_pred, _ = model(X_batch, volatility_batch, volume_batch)  # Updated this line
                loss = criterion(y_pred.squeeze(), y_batch)
                test_loss += loss.item()
                test_actuals.extend(y_batch.cpu().numpy())
                test_predictions.extend(y_pred.squeeze().cpu().numpy())
        test_loss /= len(data_loaders['test'])
        print(f'Test Loss: {test_loss:.6f}')

        # Save and display results
        logging.info("Saving and displaying results")
        save_and_display_results(test_actuals, test_predictions, subfolder)
        average_dollar_difference = evaluate_dollar_difference(model, data_loaders['test'], scaler_y, device)
        print(f'Average Dollar Difference: ${average_dollar_difference:.2f}')

        save_experiment_results(
            training_time, avg_time_per_epoch, test_loss, average_dollar_difference,
            config.get('data_limit', 'N/A'), config.get('use_pca', False), csv_path
        )

        save_and_display_results(test_actuals, test_predictions, subfolder)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        traceback.print_exc()

    logging.info("Main completed")


if __name__ == "__main__":
    config_path = '../../config/config.yaml'
    main(config_path)
