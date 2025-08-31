import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# =======================
# PyTorch Modelleri
# =======================

class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden_layer_size: int = 100, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, input_size)
        out, _ = self.lstm(x)            # (B, seq_len, H)
        out = out[:, -1, :]              # (B, H)
        out = self.linear(out)           # (B, 1)
        return out


class GRUModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden_layer_size: int = 100, output_size: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.linear(out)
        return out


# =======================
# Yardımcı Fonksiyonlar
# =======================

def create_dataset(series_scaled: np.ndarray, look_back: int = 60):
    """
    Tek değişkenli zaman serisini (N, 1) alır ve
    (num_samples, look_back, 1) X ile (num_samples,) y üretir.
    """
    X, y = [], []
    for i in range(len(series_scaled) - look_back - 1):
        X.append(series_scaled[i:i + look_back, 0])   # (look_back,)
        y.append(series_scaled[i + look_back, 0])     # scalar
    X = np.array(X).reshape(-1, look_back, 1)         # (N, look_back, 1)
    y = np.array(y).reshape(-1)                       # (N,)
    return X, y


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).float()


def train_model(model: nn.Module, train_X: np.ndarray, train_Y: np.ndarray, epochs: int = 10, lr: float = 1e-3):
    """
    Vektörize eğitim (küçük/orta veri için hızlı ve stabil).
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X = _to_tensor(train_X)                 # (N, look_back, 1)
    y = _to_tensor(train_Y).unsqueeze(1)    # (N, 1)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)                   # (N, 1)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

    return float(loss.item())


# =======================
# Ana Pipeline
# =======================

def run_prediction_pipeline(ticker_symbol: str, look_back: int = 60, epochs: int = 10):
    """
    1) Veri çek
    2) MinMax ölçekle
    3) Train/Test ayır
    4) LSTM ve GRU eğit
    5) Eğitim & test tahminlerini üret
    6) Tahminleri orijinal skalaya ve zaman eksenine yerleştir
    7) Görselleştirme için DataFrame döndür
    """
    # 1) Veri
    data = yf.download(ticker_symbol, period="3y", progress=False)
    if data.empty:
        raise ValueError(f"{ticker_symbol} için veri çekilemedi.")

    # Plotly’de timezone kaynaklı sorun yaşamamak için
    if hasattr(data.index, "tz") and data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    close = data["Close"].astype(float).values.reshape(-1, 1)

    # 2) Ölçekleme
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close)  # (N, 1)

    # 3) Train/Test
    train_size = int(len(scaled) * 0.80)
    train, test = scaled[:train_size, :], scaled[train_size:, :]

    # 4) Dataset
    train_X, train_Y = create_dataset(train, look_back=look_back)
    test_X, test_Y = create_dataset(test, look_back=look_back)

    # 5) Modeller
    lstm = LSTMModel(input_size=1)
    gru = GRUModel(input_size=1)

    lstm_loss = train_model(lstm, train_X, train_Y, epochs=epochs)
    gru_loss = train_model(gru, train_X, train_Y, epochs=epochs)

    # 6) Tahminler
    lstm.eval()
    gru.eval()
    with torch.no_grad():
        lstm_train_pred = lstm(_to_tensor(train_X)).numpy()  # (N_train, 1)
        lstm_test_pred  = lstm(_to_tensor(test_X)).numpy()   # (N_test, 1)
        gru_train_pred  = gru(_to_tensor(train_X)).numpy()
        gru_test_pred   = gru(_to_tensor(test_X)).numpy()

    # 7) Orijinal skala
    lstm_train_pred = scaler.inverse_transform(lstm_train_pred)
    lstm_test_pred  = scaler.inverse_transform(lstm_test_pred)
    gru_train_pred  = scaler.inverse_transform(gru_train_pred)
    gru_test_pred   = scaler.inverse_transform(gru_test_pred)

    # 8) Grafik için hizalama (1D float seriler)
    results_df = data.copy()
    N = len(close)

    # LSTM
    train_plot_lstm = np.full(N, np.nan, dtype="float64")
    train_plot_lstm[look_back:look_back + len(lstm_train_pred)] = lstm_train_pred.ravel()

    test_plot_lstm = np.full(N, np.nan, dtype="float64")
    test_start = len(lstm_train_pred) + (look_back * 2) + 1
    test_end = min(test_start + len(lstm_test_pred), N)
    if test_start < N:
        test_plot_lstm[test_start:test_end] = lstm_test_pred.ravel()[: (test_end - test_start)]

    # GRU
    train_plot_gru = np.full(N, np.nan, dtype="float64")
    train_plot_gru[look_back:look_back + len(gru_train_pred)] = gru_train_pred.ravel()

    test_plot_gru = np.full(N, np.nan, dtype="float64")
    test_start_g = len(gru_train_pred) + (look_back * 2) + 1
    test_end_g = min(test_start_g + len(gru_test_pred), N)
    if test_start_g < N:
        test_plot_gru[test_start_g:test_end_g] = gru_test_pred.ravel()[: (test_end_g - test_start_g)]

    # 9) Sonuç DF (1D float kolonlar)
    results_df["Close"] = results_df["Close"].astype("float64")
    results_df["LSTM_Eğitim_Tahmin"] = pd.Series(train_plot_lstm, index=results_df.index, dtype="float64")
    results_df["LSTM_Test_Tahmin"]   = pd.Series(test_plot_lstm,  index=results_df.index, dtype="float64")
    results_df["GRU_Eğitim_Tahmin"]  = pd.Series(train_plot_gru,  index=results_df.index, dtype="float64")
    results_df["GRU_Test_Tahmin"]    = pd.Series(test_plot_gru,   index=results_df.index, dtype="float64")

    return results_df, lstm_loss, gru_loss
