import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from model_utils import load_prices, ts_train_test_split, train_eval, baseline_naive

st.set_page_config(
    page_title="Karşılaştırmalı Model Eğitimi",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align:center;'>🤖 LSTM vs GRU Karşılaştırması</h1>", unsafe_allow_html=True)
st.caption("Aynı veri ayrımı üzerinde baseline, LSTM ve GRU performanslarını kıyasla.")

# --- Sembol & Parametreler ---
tickers = {
    "Akbank (AKBNK.IS)": "AKBNK.IS",
    "Aselsan (ASELS.IS)": "ASELS.IS",
    "THY (THYAO.IS)": "THYAO.IS",
    "Garanti (GARAN.IS)": "GARAN.IS",
    "BIST100 (XU100.IS)": "XU100.IS",
    "Apple (AAPL)": "AAPL"
}

c1, c2, c3, c4 = st.columns(4)
with c1:
    name = st.selectbox("Sembol", list(tickers.keys()))
with c2:
    win = st.number_input("Pencere (gün)", min_value=20, max_value=120, value=30, step=5)
with c3:
    epochs = st.number_input("Epoch", min_value=5, max_value=100, value=20, step=5)
with c4:
    hidden = st.number_input("Hidden size", min_value=16, max_value=256, value=64, step=16)

c5, c6 = st.columns(2)
with c5:
    start = st.date_input("Başlangıç", pd.Timestamp.today() - pd.DateOffset(years=3))
with c6:
    end   = st.date_input("Bitiş", pd.Timestamp.today())

run = st.button("Eğit ve Karşılaştır", type="primary", use_container_width=True)
st.markdown("---")

if run:
    with st.spinner("Veri çekiliyor ve modeller eğitiliyor..."):
        df = load_prices(tickers[name], start=start, end=end)

        if len(df) < (win * 2 + 30):
            st.warning("Veri kısa görünüyor. Daha uzun bir tarih aralığı seçin.")
        else:
            # ---- split ve eğitim ----
            train_s, test_s, split_idx = ts_train_test_split(df['close'], test_ratio=0.2)

            b_pred, b_true, b_rmse, b_mae, b_mape = baseline_naive(test_s.values)

            lstm_pred, y_true, l_rmse, l_mae, l_mape = train_eval(
                kind='lstm',
                train_vals=train_s.values,
                test_vals=test_s.values,
                win=win, epochs=epochs, hidden=hidden
            )

            gru_pred, _, g_rmse, g_mae, g_mape = train_eval(
                kind='gru',
                train_vals=train_s.values,
                test_vals=test_s.values,
                win=win, epochs=epochs, hidden=hidden
            )

            # --------- Grafik: gerçek vs LSTM vs GRU + split çizgisi ---------
            fig = go.Figure()
            idx = df.index
            train_idx = idx[:split_idx]
            test_idx  = idx[split_idx:]

            # Gerçek değerler
            fig.add_trace(go.Scatter(x=train_idx, y=df['close'].iloc[:split_idx],
                                     mode="lines", name="Gerçek (Train)"))
            fig.add_trace(go.Scatter(x=test_idx,  y=df['close'].iloc[split_idx:],
                                     mode="lines", name="Gerçek (Test)"))

            # Train/Test ayrım çizgisi — add_vline yerine shape + annotation (datetime ile sorunsuz)
            if len(train_idx) > 0:
                split_x = pd.to_datetime(train_idx[-1])
                fig.add_shape(
                    type="line",
                    x0=split_x, x1=split_x, y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(dash="dash", width=1)
                )
                fig.add_annotation(
                    x=split_x, y=1, xref="x", yref="paper",
                    text="Train/Test Split", showarrow=False, yanchor="bottom"
                )

            # Baseline, LSTM ve GRU tahminleri
            fig.add_trace(go.Scatter(x=test_idx[1:], y=b_pred,
                                     mode="lines", name="Naive (t-1)"))
            fig.add_trace(go.Scatter(x=test_idx, y=lstm_pred,
                                     mode="lines", name="LSTM"))
            fig.add_trace(go.Scatter(x=test_idx, y=gru_pred,
                                     mode="lines", name="GRU"))

            fig.update_layout(
                title=f"{name} — LSTM vs GRU",
                yaxis_title="Fiyat",
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", y=1.12, x=1, xanchor="right")
            )
            st.plotly_chart(fig, use_container_width=True)

            # --------- Metrik Tablosu ---------
            metrics = pd.DataFrame([
                ["Naive (t-1)", b_rmse, b_mae, b_mape],
                ["LSTM",        l_rmse, l_mae, l_mape],
                ["GRU",         g_rmse, g_mae, g_mape],
            ], columns=["Model", "RMSE", "MAE", "MAPE (%)"])

            st.subheader("Test Dönemi Metrikleri")
            st.dataframe(
                metrics.style.format({"RMSE": "{:.3f}", "MAE": "{:.3f}", "MAPE (%)": "{:.2f}"}),
                use_container_width=True
            )

            best = metrics.sort_values("RMSE").iloc[0]
            st.info(f"En düşük RMSE: **{best['Model']}** — Sunumda grafiği ve bu tabloyu kullan.")
