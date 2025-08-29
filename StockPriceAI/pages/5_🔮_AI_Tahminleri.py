# pages/3_🔮_AI_Tahminleri.py

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

# --- Sayfa Ayarları ---
st.set_page_config(layout="wide")
st.title("🔮 AI Tahminleri & Teknik Analiz")

# --- Model Oluşturma Fonksiyonu ---
def create_model(model_type, input_shape):
    """Kullanıcının seçimine göre LSTM veya GRU modeli oluşturur."""
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50, return_sequences=False))
    elif model_type == 'GRU':
        model.add(GRU(50, return_sequences=True, input_shape=input_shape))
        model.add(GRU(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- Kenar Çubuğu (Sidebar) ---
with st.sidebar:
    st.header("Tahmin Ayarları")
    
    hisse_secimi = st.selectbox(
        "Hisse Seçin:",
        ("AAPL", "MSFT", "GOOGL", "THYAO.IS", "GARAN.IS"),
        key="tahmin_hisse_secimi"
    )

    tahmin_gunu = st.slider(
        "Tahmin Günü Sayısı", min_value=1, max_value=30, value=7, step=1
    )

    model_secimi = st.selectbox(
        "Model Seçimi",
        ("LSTM", "GRU"), # Ensemble kaldırıldı
        key="tahmin_model_secimi"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <style>
        div.stButton > button { background-color: #FF4B4B; color: white; font-weight: bold; border-radius: 8px; border: none; padding: 10px 0; width: 100%; }
        div.stButton > button:hover { background-color: #E03C3C; color: white; }
    </style>
    """, unsafe_allow_html=True)

    run_button = st.button("📈 Veri Yükle & Tahmin Et")

# --- Ana Ekran ---

if run_button:
    with st.spinner(f"{model_secimi} modeli {hisse_secimi} için eğitiliyor ve tahminler oluşturuluyor..."):
        try:
            # 1. Veri Çekme ve Hazırlama
            data = yf.download(hisse_secimi, start="2020-01-01", end=pd.Timestamp.now())
            if len(data) < 61:
                st.warning("Model eğitimi için yeterli veri yok (en az 61 gün gerekli).")
                st.stop()
            
            # Veri sütun adlarını standart hale getir
            data.columns = [str(col).lower() for col in data.columns]
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))
            
            X_train, y_train = [], []
            for i in range(60, len(scaled_data)):
                X_train.append(scaled_data[i-60:i, 0])
                y_train.append(scaled_data[i, 0])
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

            # 2. Modeli Eğitme
            model = create_model(model_secimi, (X_train.shape[1], 1))
            model.fit(X_train, y_train, batch_size=32, epochs=25, verbose=0)
            
            # 3. Geçmişe Yönelik Tahminleri Oluşturma
            past_predictions_scaled = model.predict(X_train)
            past_predictions = scaler.inverse_transform(past_predictions_scaled)

            # 4. Geleceğe Yönelik Tahminleri Oluşturma
            last_60_days = scaled_data[-60:]
            current_batch = last_60_days.reshape(1, 60, 1)
            future_predictions = []
            for i in range(tahmin_gunu):
                next_prediction_scaled = model.predict(current_batch)
                future_predictions.append(next_prediction_scaled[0, 0])
                next_prediction_reshaped = next_prediction_scaled.reshape(1, 1, 1)
                current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)
            future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
            
            # --- Sonuçları Göster ---
            st.success("Tahminler başarıyla oluşturuldu!")
            
            tab1, tab2, tab3 = st.tabs(["Fiyat Tahminleri", "Teknik Sinyaller", "Model Analizi"])
            with tab1:
                st.subheader("Tahmin Özeti")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Ortalama Tahmin", f"{future_predictions_rescaled.mean():.2f}")
                col2.metric("Maksimum Fiyat", f"{future_predictions_rescaled.max():.2f}")
                col3.metric("Minimum Fiyat", f"{future_predictions_rescaled.min():.2f}")
                
                # Hata düzeltmesi: last_price'ın tek bir değer olduğundan emin ol
                last_price_value = data['close'].iloc[-1]
                volatility = future_predictions_rescaled.std()
                risk = "Düşük" if volatility < (last_price_value * 0.02) else "Orta" if volatility < (last_price_value * 0.05) else "Yüksek"
                col4.metric("Risk Seviyesi", risk, f"Vol: {volatility:.2f}")

                st.subheader("Fiyat Tahmin Grafiği")
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(x=data.index, y=data['close'], mode='lines', name='Geçmiş Fiyatlar', line=dict(color='blue')))
                
                prediction_dates = data.index[60:]
                fig.add_trace(go.Scatter(x=prediction_dates, y=past_predictions.flatten(), mode='lines', name='Geçmiş AI Tahminleri', line=dict(color='orange')))

                last_known_date = data.index[-1]
                future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=tahmin_gunu)
                fig.add_trace(go.Scatter(x=future_dates, y=future_predictions_rescaled, mode='lines', name='Gelecek AI Tahminleri', line=dict(color='red', dash='dot')))
                
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

else:
    st.info("Lütfen kenar çubuğundan ayarları seçip 'Veri Yükle & Tahmin Et' butonuna basın.")

