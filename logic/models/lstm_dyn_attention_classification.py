import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from ta import add_all_ta_features
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import logging
import traceback
import torch.optim.lr_scheduler as lr_scheduler

from logic.models.abstract_model import set_up_folders, save_experiment_results
from src.data_preprocessing.data_importer import import_data
from src.utils.config_loader import load_config
from src.utils.data_saving_and_displaying import save_and_display_results, save_and_display_results_classification
from src.data_preprocessing.data_preprocessor import DataPreprocessor

from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

import time

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

train_processed, train_volatility, train_volume = None, None, None
val_processed, val_volatility, val_volume = None, None, None
test_processed, test_volatility, test_volume = None, None, None


class DynamicAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(DynamicAttention, self).__init__()
        self.feature_layer = nn.Linear(2, hidden_dim, bias=False)
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out, volatility, volume):

        features = torch.cat((volatility.unsqueeze(-1), volume.unsqueeze(-1)), dim=-1)
        dynamic_weights = torch.tanh(self.feature_layer(features))
        attention_weights = torch.softmax(self.attention(lstm_out * dynamic_weights).squeeze(-1), dim=1)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * lstm_out, dim=1)
        return context_vector, attention_weights


class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.0, use_attention=True):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        self.attention = DynamicAttention(hidden_dim)

        self.fc_layers = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim // 2),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim // 4),

            nn.Linear(hidden_dim // 4, num_classes)
        )
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x, volatility, volume):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))


        if self.use_attention:
            context_vector, attention_weights = self.attention(lstm_out, volatility, volume)
        else:
            seq_len = lstm_out.size(1)
            uniform_attention_weights = torch.ones(lstm_out.size(0), seq_len, device=lstm_out.device) / seq_len

            context_vector = torch.sum(uniform_attention_weights.unsqueeze(-1) * lstm_out, dim=1)
            attention_weights = uniform_attention_weights

        out = self.fc_layers(context_vector)
        #out = self.softmax(out)
        return out, attention_weights


class CryptoDataset(Dataset):
    def __init__(self, data, volatility, volume, seq_length, absolute_prices):
        self.data = torch.FloatTensor(data[:, :-1])
        self.volatility = torch.FloatTensor(volatility)
        self.volume = torch.FloatTensor(volume)
        self.seq_length = seq_length
        self.labels = torch.LongTensor(data[:, -1].astype(int))
        self.absolute_prices = torch.FloatTensor(absolute_prices)

        expected_length = len(self)
        if len(self.labels) > expected_length:
            self.labels = self.labels[-expected_length:]
            self.absolute_prices = self.absolute_prices[-expected_length:]

    def __len__(self):
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, idx):
        return (self.data[idx:idx + self.seq_length],
                self.volatility[idx:idx + self.seq_length],
                self.volume[idx:idx + self.seq_length],
                self.labels[idx],
                self.absolute_prices[idx])

def add_custom_ta_features(data):
    # MACD
    macd = MACD(close=data['Close'])
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_diff'] = macd.macd_diff()

    # EMA
    data['ema_9'] = EMAIndicator(close=data['Close'], window=9).ema_indicator()
    data['ema_21'] = EMAIndicator(close=data['Close'], window=21).ema_indicator()
    data['ema_50'] = EMAIndicator(close=data['Close'], window=50).ema_indicator()
    data['ema_200'] = EMAIndicator(close=data['Close'], window=200).ema_indicator()

    # RSI
    data['rsi_14'] = RSIIndicator(close=data['Close'], window=14).rsi()
    data['rsi_21'] = RSIIndicator(close=data['Close'], window=21).rsi()

    # Bollinger Bands
    bb = BollingerBands(close=data['Close'])
    data['bb_high'] = bb.bollinger_hband()
    data['bb_low'] = bb.bollinger_lband()
    data['bb_mid'] = bb.bollinger_mavg()
    data['bb_width'] = (data['bb_high'] - data['bb_low']) / data['bb_mid']

    # On-Balance Volume
    data['obv'] = OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()

    # Price rate of change
    data['price_roc'] = data['Close'].pct_change(periods=12)

    return data


def calculate_volatility(data, window_size=20):
    data['log_return'] = np.log(data['Close']) - np.log(data['Close'].shift(1))
    data['volatility'] = data['log_return'].rolling(window=window_size).std()
    return data['volatility'].dropna()

def compute_grad_norms(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def aggregate_and_save_cv_results(cv_results, subfolder):
    all_test_actuals = []
    all_test_predictions = []
    avg_test_loss = 0

    for result in cv_results:
        all_test_actuals.extend(result['test_actuals'])
        all_test_predictions.extend(result['test_predictions'])
        avg_test_loss += result['test_loss']

    avg_test_loss /= len(cv_results)

    save_and_display_results_classification(all_test_actuals, all_test_predictions, subfolder,
                                            dataset='test_aggregated')

    with open(os.path.join(subfolder, 'cv_results.txt'), 'w') as f:
        f.write(f"Average Test Loss: {avg_test_loss:.6f}\n")

    logging.info(f"Aggregated cross-validation results saved. Average Test Loss: {avg_test_loss:.6f}")


def preprocess_data(data: pd.DataFrame, config, data_preprocessor: DataPreprocessor):
    target = config['target']

    data = data.copy()

    data['Close_diff'] = data['Close'].pct_change()

    data['target'] = data[target].shift(-1)

    data = data.dropna().reset_index(drop=True)

    # data = add_custom_ta_features(data)
    # data = data.dropna().reset_index(drop=True)
    #
    # feature_columns = ['macd', 'macd_signal', 'macd_diff',
    #                    'ema_9', 'ema_21', 'ema_50', 'ema_200',
    #                    'rsi_14', 'rsi_21',
    #                    'bb_high', 'bb_low', 'bb_mid', 'bb_width',
    #                    'obv', 'price_roc']
    absolute_prices = data['Close'].values

    data = add_all_ta_features(data, "Open", "High", "Low", "Close", "Volume", fillna=True)
    data = data.dropna().reset_index(drop=True)

    look_ahead_indicators = ['trend_ichimoku_a', 'trend_ichimoku_b', 'trend_visual_ichimoku_a',
                             'trend_visual_ichimoku_b', 'trend_stc', 'trend_psar_up', 'trend_psar_down']

    feature_columns = [col for col in data.columns if col not in (
                ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'target',
                 'Average_Close_diff'] + look_ahead_indicators)]

    logging.info(f"Number of features before PCA: {len(feature_columns)}")

    data['volatility'] = calculate_volatility(data, window_size=config.get('volatility_window_size', 20))

    data = data.drop(columns=['Close'])

    data = data.dropna().reset_index(drop=True)

    volatility = data['volatility']
    volume = data['Volume']

    X = data[feature_columns].values
    y = data['target'].values

    X_scaled, y_scaled, volatility_scaled, volume_scaled = data_preprocessor.fit_transform_data(X, y, volatility, volume,
                                                                                                subfolder)

    logging.info(f"Number of features after preprocessing: {X_scaled.shape[1]}")

    assert not np.isnan(X_scaled).any(), "NaN values found in features"
    assert not np.isnan(y_scaled).any(), "NaN values found in target"
    assert not np.isnan(volatility_scaled).any(), "NaN values found in volatility"

    return np.hstack((X_scaled, y_scaled.reshape(-1, 1))), volatility_scaled, volume_scaled, absolute_prices


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience=5):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, volatility_batch, volume_batch, y_batch, price_batch in train_loader:
            X_batch, volatility_batch, volume_batch, y_batch = X_batch.to(device), volatility_batch.to(device), volume_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred, _ = model(X_batch, volatility_batch, volume_batch)

            assert not torch.isnan(y_pred).any(), "NaN values found in model output"
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        grad_norm = compute_grad_norms(model)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, volatility_batch, volume_batch, y_batch, price_batch in val_loader:
                X_batch, volatility_batch, volume_batch, y_batch = X_batch.to(device), volatility_batch.to(device), volume_batch.to(device), y_batch.to(device)
                y_pred, _ = model(X_batch, volatility_batch, volume_batch)

                assert not torch.isnan(y_pred).any(), "NaN values found in model output"
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Grad Norm: {grad_norm:.6f}')

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

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    actuals = []
    predictions = []
    absolute_prices = []
    with torch.no_grad():
        for inputs, volatility, volume, targets, prices in data_loader:
            inputs, volatility, volume, targets = inputs.to(device), volatility.to(device), volume.to(device), targets.to(device)
            outputs, _ = model(inputs, volatility, volume)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            actuals.extend(targets.cpu().numpy())
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            absolute_prices.extend(prices.cpu().numpy())
    return total_loss / len(data_loader), actuals, predictions, absolute_prices


def save_results_with_prices(actuals, predictions, absolute_prices, subfolder, dataset):
    results_df = pd.DataFrame({
        'Actual': actuals,
        'Predicted': predictions,
        'AbsolutePrice': absolute_prices
    })
    results_df.to_csv(os.path.join(subfolder, f'{dataset}_results_with_prices.csv'), index=False)
    logging.info(f"Results with absolute prices saved to {dataset}_results_with_prices.csv")


def evaluate_dollar_difference(model, data_loader, scaler_y, device):
    model.eval()
    total_abs_error = 0
    count = 0

    if not isinstance(scaler_y, StandardScaler):
        raise TypeError(f"Expected StandardScaler, but got {type(scaler_y)}")

    with torch.no_grad():
        for X_batch, volatility_batch, volume_batch, y_batch in data_loader:
            X_batch, volatility_batch, volume_batch, y_batch = X_batch.to(device), volatility_batch.to(
                device), volume_batch.to(device), y_batch.to(device)
            y_pred, _ = model(X_batch, volatility_batch, volume_batch)

            y_pred = y_pred[-len(y_batch):, :]

            y_pred_np = y_pred.cpu().numpy()
            y_batch_np = y_batch.cpu().numpy().reshape(-1, 1)

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


def setup_logging(subfolder):
    log_filename = os.path.join(subfolder, f'experiment_log_{time.strftime("%Y%m%d-%H%M")}.log')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    return log_filename


def main(config_path, grid_search_run=None):
    config = load_config(config_path)
    global subfolder, train_processed, train_volatility, test_volume, test_volatility, test_processed, val_volume, val_volatility, val_processed, train_volume
    global device
    global project_root

    project_root, subfolder = set_up_folders()

    if grid_search_run is not None:
        subfolder = os.path.join(subfolder, f'grid_search_{grid_search_run}')
        os.makedirs(subfolder, exist_ok=True)

    log_filename = setup_logging(subfolder)
    logging.info(f"Logging to file: {log_filename}")

    csv_path = os.path.join(subfolder, 'times.csv')

    try:
        logging.info("Starting main function")
        logging.info(f"Configuration loaded from: {config_path}")
        logging.info(f"Using device: {device}")

        training_time = 0
        avg_time_per_epoch = 0

        all_data_filenames = config['train_data'] + config['val_data'] + config['test_data']
        all_data_paths = [os.path.join(project_root, 'data', path) for path in all_data_filenames]
        all_data = import_data(all_data_paths, limit=config.get('data_limit', None))
        all_data = all_data.sort_values('date').reset_index(drop=True)

        scaler_type = config.get('scaler', 'standard')
        use_pca = config.get('use_pca', False)

        test_split = int(0.8 * len(all_data))
        train_val_data = all_data[:test_split]
        test_data = all_data[test_split:]

        val_split = int(0.8 * len(train_val_data))
        train_data = train_val_data[:val_split]
        val_data = train_val_data[val_split:]

        logging.info(f"Data split - Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

        data_preprocessor = DataPreprocessor(scaler_type=scaler_type, use_pca=use_pca)

        if train_processed is None:
            train_processed, train_volatility, train_volume, train_prices = preprocess_data(train_data, config,
                                                                                            data_preprocessor)
            val_processed, val_volatility, val_volume, val_prices = preprocess_data(val_data, config, data_preprocessor)
            test_processed, test_volatility, test_volume, test_prices = preprocess_data(test_data, config,
                                                                                        data_preprocessor)

        train_dataset = CryptoDataset(train_processed, train_volatility, train_volume, config['seq_length'],
                                      train_prices)
        val_dataset = CryptoDataset(val_processed, val_volatility, val_volume, config['seq_length'], val_prices)
        test_dataset = CryptoDataset(test_processed, test_volatility, test_volume, config['seq_length'], test_prices)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], drop_last=True)

        input_dim = train_processed.shape[1] - 1
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
        dropout = config.get('dropout', 0)
        use_attention = config['use_attention']
        model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=3,
                          dropout=dropout, use_attention=use_attention)
        logging.info(f"Model initialized with hidden_dim: {hidden_dim}, num_layers: {num_layers}, dropout: {dropout}")

        criterion = CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                     weight_decay=config.get('weight_decay', 0))
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        logging.info("Starting model training")
        start_time = time.time()
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config['num_epochs'])
        end_time = time.time()
        training_time = end_time - start_time
        avg_time_per_epoch = training_time / config['num_epochs']
        logging.info(
            f"Model training completed. Total time: {training_time:.2f}s, Avg time per epoch: {avg_time_per_epoch:.2f}s")

        logging.info("Starting model evaluation on test set")
        test_loss, test_actuals, test_predictions, test_prices = evaluate_model(model, test_loader, criterion, device)
        logging.info(f"Test evaluation completed. Test loss: {test_loss:.4f}")

        save_results_with_prices(test_actuals, test_predictions, test_prices, subfolder, dataset=f'test_pca_{use_pca}')

        save_and_display_results_classification(test_actuals, test_predictions, subfolder, dataset=f'test_pca_{use_pca}')
        logging.info("Results saved and displayed")

        save_experiment_results(training_time, avg_time_per_epoch, test_loss, 0.0, config.get('data_limit', 'N/A'),
                                config.get('use_pca', False), csv_path)
        logging.info("Experiment results saved")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc())

    logging.info("Main completed")


if __name__ == "__main__":
    config_path = '../../config/config.yaml'
    main(config_path)