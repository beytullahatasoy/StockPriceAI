import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from model_utils import run_prediction_pipeline

st.set_page_config(page_title="Karşılaştırmalı Model Eğitimi", layout="wide")

# --- Başlık & Açıklama ---
st.markdown("<h1 style='text-align:center;'>🤖 PyTorch ile LSTM ve GRU Karşılaştırması</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:grey;'>Seçtiğiniz hisse için LSTM ve GRU modellerini eğitin, test tahminlerini görselleştirin.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Hisse Listesi ---
bist30_tickers = {
    "Akbank (AKBNK.IS)": "AKBNK.IS", "Arçelik (ARCLK.IS)": "ARCLK.IS", "Aselsan (ASELS.IS)": "ASELS.IS",
    "BİM Mağazalar (BIMAS.IS)": "BIMAS.IS", "Ereğli Demir Çelik (EREGL.IS)": "EREGL.IS", "Ford Otosan (FROTO.IS)": "FROTO.IS",
    "Garanti Bankası (GARAN.IS)": "GARAN.IS", "Koç Holding (KCHOL.IS)": "KCHOL.IS", "Pegasus (PGSUS.IS)": "PGSUS.IS",
    "Sabancı Holding (SAHOL.IS)": "SAHOL.IS", "Şişecam (SISE.IS)": "SISE.IS", "Turkcell (TCELL.IS)": "TCELL.IS",
    "Türk Hava Yolları (THYAO.IS)": "THYAO.IS", "Tüpraş (TUPRS.IS)": "TUPRS.IS", "Yapı Kredi (YKBNK.IS)": "YKBNK.IS"
}

# --- Girdiler ---
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    selected_name = st.selectbox("Analiz edilecek hisse:", options=list(bist30_tickers.keys()))
with col2:
    epoch_count = st.slider("Epoch", min_value=5, max_value=50, step=5, value=15,
                            help="Epoch arttıkça öğrenme artar, süre uzar.")
with col3:
    look_back = st.slider("Look-back (seq_len)", min_value=20, max_value=120, step=10, value=60,
                          help="Girdi penceresi uzunluğu.")

ticker = bist30_tickers[selected_name]

if st.button(f"📈 {epoch_count} Epoch ile Eğit & Analiz Başlat", use_container_width=True, type="primary"):
    try:
        with st.spinner(f"'{selected_name}' için eğitim ve tahmin çalışıyor..."):
            results_df, lstm_loss, gru_loss = run_prediction_pipeline(
                ticker, look_back=look_back, epochs=epoch_count
            )

        st.success("✅ Analiz başarıyla tamamlandı!")

        # --- Görselleştirme ---
        st.subheader(f"{selected_name} - Model Tahmin Grafiği")

        # Tipleri garanti et
        results_df["Close"] = pd.to_numeric(results_df["Close"], errors="coerce")
        for c in ["LSTM_Test_Tahmin", "GRU_Test_Tahmin"]:
            if c in results_df.columns:
                results_df[c] = pd.to_numeric(results_df[c], errors="coerce")

        close_series = results_df["Close"]
        lstm_test = results_df["LSTM_Test_Tahmin"].dropna()
        gru_test  = results_df["GRU_Test_Tahmin"].dropna()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results_df.index, y=close_series.values,
                                 mode="lines", name="Gerçek Fiyat"))
        if not lstm_test.empty:
            fig.add_trace(go.Scatter(x=lstm_test.index, y=lstm_test.values,
                                     mode="lines", name="LSTM Test Tahmini"))
        if not gru_test.empty:
            fig.add_trace(go.Scatter(x=gru_test.index, y=gru_test.values,
                                     mode="lines", name="GRU Test Tahmini"))

        fig.update_layout(
            yaxis_title="Fiyat (TL)",
            xaxis_title="Tarih",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=10, b=10),
        )
        fig.update_traces(connectgaps=True)  # NaN boşluklarını bağla

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Model Performans Metrikleri")

        m1, m2 = st.columns(2)
        with m1:
            st.metric("LSTM Model Eğitim Kaybı (MSE)", f"{lstm_loss:.6f}")
            st.info("Daha düşük MSE ➜ eğitim verisine daha iyi uyum.")
        with m2:
            st.metric("GRU Model Eğitim Kaybı (MSE)", f"{gru_loss:.6f}")
            st.info("LSTM/GRU kayıplarını kıyaslayarak öğrenme kalitesini görebilirsin.")

    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")
        st.warning("İnternet bağlantını kontrol et; farklı hisse/epoch/seq_len ile tekrar dene.")
