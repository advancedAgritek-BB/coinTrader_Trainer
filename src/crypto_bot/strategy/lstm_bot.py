"""LSTM based price predictor for crypto trading bots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMModel(nn.Module):
    """Simple LSTM model for sequence forecasting.

    Parameters
    ----------
    input_size: int
        Number of features in the input sequence. Defaults to 1 (price).
    hidden_size: int
        Number of hidden units in each LSTM layer.
    num_layers: int
        Number of stacked LSTM layers.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1])
        return out


def _create_sequences(data: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Transform 1D array ``data`` into supervised learning sequences."""
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i : i + seq_len])
        ys.append(data[i + seq_len])
    return np.array(xs), np.array(ys)


def train_lstm(csv_path: str, use_gpu: bool = True) -> LSTMModel:
    """Train an :class:`LSTMModel` on closing price data from ``csv_path``.

    The function loads a CSV file containing a ``Close`` column (or the first
    numeric column if ``Close`` is absent), builds fixed-length sequences and
    trains the model using ``AdamW`` with early stopping.

    Parameters
    ----------
    csv_path: str
        Path to CSV file with price data.
    use_gpu: bool, optional
        Move model to CUDA if available. Defaults to ``True``.

    Returns
    -------
    LSTMModel
        Trained model instance.
    """

    df = pd.read_csv(csv_path)
    price_col = None
    for cand in ["Close", "close", "price"]:
        if cand in df.columns:
            price_col = cand
            break
    if price_col is None:
        price_col = df.select_dtypes(include=[np.number]).columns[0]

    prices = df[price_col].astype("float32").values

    seq_len = 50
    X, y = _create_sequences(prices, seq_len)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    X_train = torch.tensor(X_train).unsqueeze(-1)
    y_train = torch.tensor(y_train)
    X_val = torch.tensor(X_val).unsqueeze(-1)
    y_val = torch.tensor(y_val)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    best_val = float("inf")
    patience = 5
    patience_counter = 0

    for _ in range(100):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x).squeeze()
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device)
                vy = vy.to(device)
                pred = model(vx).squeeze()
                val_losses.append(criterion(pred, vy).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "lstm_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model
