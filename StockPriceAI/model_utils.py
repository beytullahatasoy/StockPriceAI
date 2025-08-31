# model_utils.py
# Yardımcı fonksiyonlar: veri çekme, pencereleme, baseline, LSTM/GRU eğitim & değerlendirme

import os, random
import numpy as np
import pandas as pd
import yfinance as yf

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ------------ Genel yardımcılar ------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prices(ticker: str, start=None, end=None, interval: str = "1d") -> pd.DataFrame:
    """
    yfinance'tan fiyat verisini indirir (Auto Adjust açık).
    Dönen df: ['close'] kolonu (NaN'ler atılmış).
    """
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if 'Close' not in df.columns:
        raise ValueError(f"{ticker} için 'Close' kolonu bulunamadı.")
    return df[['Close']].dropna().rename(columns={'Close': 'close'})


def ts_train_test_split(series: pd.Series, test_ratio=0.2):
    """
    Zaman serisini sırayı bozmadan train/test'e ayırır.
    """
    n = len(series)
    split_idx = max(30, int(n * (1 - test_ratio)))
    return series.iloc[:split_idx].copy(), series.iloc[split_idx:].copy(), split_idx


def make_windows(arr: np.ndarray, win: int = 30, horizon: int = 1):
    """
    Kaydırmalı pencere (X: [win], y: gelecek horizon).
    """
    X, y = [], []
    for i in range(win, len(arr) - horizon + 1):
        X.append(arr[i - win:i])
        y.append(arr[i + horizon - 1])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


# ------------ Modeller ------------
class RNNModel(nn.Module):
    def __init__(self, kind='lstm', input_size=1, hidden=64, layers=2, dropout=0.0):
        super().__init__()
        rnn = nn.LSTM if kind.lower() == 'lstm' else nn.GRU
        self.rnn = rnn(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            dropout=(dropout if layers > 1 else 0.0),
            batch_first=True
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)          # (B, T, H)
        out = out[:, -1, :]           # (B, H) son zaman adımı
        return self.fc(out)           # (B, 1)


def train_eval(
    kind: str,
    train_vals: np.ndarray,
    test_vals: np.ndarray,
    win: int = 30,
    epochs: int = 20,
    hidden: int = 64,
    layers: int = 2,
    lr: float = 1e-3,
    device: str | None = None
):
    """
    LSTM/GRU eğitir ve TEST dönemi tahminleri ile metrikleri döndürür.
    return: (inv_preds, inv_y, rmse, mae, mape)
    """
    # --- seed & device ---
    set_seed(42)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # --- ölçekleme: sadece train'e fit ---
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1)).ravel()
    test_scaled  = scaler.transform(test_vals.reshape(-1, 1)).ravel()

    # --- pencereleme ---
    Xtr, ytr = make_windows(train_scaled, win=win)
    # test pencereleri için train'in son win kısmını ekleyerek süreklilik sağla
    concat = np.concatenate([train_scaled[-win:], test_scaled])
    Xte, yte = make_windows(concat, win=win)

    # tensörler
    Xtr = torch.tensor(Xtr[..., None]).to(device)  # (B, T, 1)
    ytr = torch.tensor(ytr[..., None]).to(device)  # (B, 1)
    Xte = torch.tensor(Xte[..., None]).to(device)
    yte = torch.tensor(yte[..., None]).to(device)

    # --- model & optim ---
    model = RNNModel(kind=kind, hidden=hidden, layers=layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # --- eğitim ---
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xtr)
        loss = loss_fn(pred, ytr)
        loss.backward()
        opt.step()

    # --- değerlendirme (test) ---
    model.eval()
    with torch.no_grad():
        preds_te = model(Xte).cpu().numpy().ravel()

    # --- inverse scale ---
    inv_preds = scaler.inverse_transform(preds_te.reshape(-1, 1)).ravel()
    inv_y     = scaler.inverse_transform(yte.cpu().numpy().reshape(-1, 1)).ravel()

    # --- metrikler (RMSE = sqrt(MSE)) ---
    mse  = mean_squared_error(inv_y, inv_preds)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(inv_y, inv_preds)
    mape = np.mean(np.abs((inv_y - inv_preds) / np.clip(np.abs(inv_y), 1e-8, None))) * 100

    return inv_preds, inv_y, rmse, mae, mape


def baseline_naive(test_vals: np.ndarray):
    """
    Naive baseline: y_hat_t = y_{t-1}
    test_vals: test dönemindeki gerçek kapanış değerleri (1D array)
    Döndürür: (y_pred, y_true, rmse, mae, mape)
    """
    y_true = test_vals[1:]
    y_pred = test_vals[:-1]

    mse  = mean_squared_error(y_true, y_pred)   # eski sklearn sürümleriyle uyumlu
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

    return y_pred, y_true, rmse, mae, mape
