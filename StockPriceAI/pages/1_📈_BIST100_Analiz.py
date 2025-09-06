import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import pandas_ta as ta
import numpy as np

# --- Sayfa Başlığı (Ortalandı) ---
st.markdown("<h1 style='text-align: center;'>📈 BIST100 Hisse Senedi Analizi</h1>", unsafe_allow_html=True)


# --- Hisse Senedi Listesi ---
bist30_tickers = {
    "Akbank (AKBNK.IS)": "AKBNK.IS",
    "Arçelik (ARCLK.IS)": "ARCLK.IS",
    "Aselsan (ASELS.IS)": "ASELS.IS",
    "BİM Mağazalar (BIMAS.IS)": "BIMAS.IS",
    "Ereğli Demir Çelik (EREGL.IS)": "EREGL.IS",
    "Ford Otosan (FROTO.IS)": "FROTO.IS",
    "Garanti Bankası (GARAN.IS)": "GARAN.IS",
    "Koç Holding (KCHOL.IS)": "KCHOL.IS",
    "Pegasus (PGSUS.IS)": "PGSUS.IS",
    "Sabancı Holding (SAHOL.IS)": "SAHOL.IS",
    "Şişecam (SISE.IS)": "SISE.IS",
    "Turkcell (TCELL.IS)": "TCELL.IS",
    "Türk Hava Yolları (THYAO.IS)": "THYAO.IS",
    "Tüpraş (TUPRS.IS)": "TUPRS.IS",
    "Yapı Kredi Bankası (YKBNK.IS)": "YKBNK.IS"
}

# --- Oturum Durumu (Session State) ile Sayfa Akışını Yönetme ---

if 'selected_bist_stock' not in st.session_state:
    st.session_state['selected_bist_stock'] = None

if st.session_state['selected_bist_stock'] is None:
    # --- YENİ BÖLÜM: TASARIM İYİLEŞTİRMELERİ ---
    st.markdown("<h3 style='text-align: center;'>Lütfen Analiz Etmek İstediğiniz Hisseleri Seçin</h3>", unsafe_allow_html=True)
    
    # Seçim kutusunu ortalamak için sütunlar kullanıyoruz
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_stock_name = st.selectbox(
            "Bir hisse senedi seçin:",
            options=list(bist30_tickers.keys()),
            label_visibility="collapsed" # Etiketi gizle
        )
        
        if st.button("Analiz Et", use_container_width=True):
            st.session_state['selected_bist_stock'] = bist30_tickers[selected_stock_name]
            st.rerun()

    st.markdown("---")

    # --- YENİ BÖLÜM: HİSSE FİYAT ÖZET TABLOSU ---
    st.subheader("BIST30 Hisseleri Güncel Durum")
    
    @st.cache_data(ttl=600)
    def get_bist_prices():
        tickers_list = list(bist30_tickers.values())
        data = yf.download(tickers_list, period="3d")
        
        summary = []
        for ticker in tickers_list:
            try:
                stock_name = [name for name, t in bist30_tickers.items() if t == ticker][0]
                last_price = data['Close'][ticker].iloc[-1]
                prev_price = data['Close'][ticker].iloc[-2]
                change_pct = ((last_price - prev_price) / prev_price) * 100
                summary.append({
                    "Hisse": stock_name,
                    "Son Fiyat": f"{last_price:.2f}",
                    "Günlük %": f"{change_pct:+.2f}%"
                })
            except (KeyError, IndexError):
                # Bazı hisseler için veri gelmezse atla
                continue
        return pd.DataFrame(summary)

    price_summary_df = get_bist_prices()
    
    # Stil fonksiyonu: Değişim pozitifse yeşil, negatifse kırmızı yap
    def color_change(val):
        color = 'red' if isinstance(val, str) and val.startswith('-') else 'green' if isinstance(val, str) and val.startswith('+') else 'white'
        return f'color: {color}'

    st.dataframe(
        price_summary_df.style.applymap(color_change, subset=['Günlük %']),
        use_container_width=True,
        hide_index=True
    )
    # --- YENİ BÖLÜM SONU ---


else:
    ticker_symbol = st.session_state['selected_bist_stock']
    
    if st.sidebar.button("↩️ Başka Hisse Seç"):
        st.session_state['selected_bist_stock'] = None
        st.rerun()

    st.sidebar.header("Zaman Aralığı")
    time_range = st.sidebar.radio("Zaman Aralığı Seçin:", ("Son 1 Ay", "Son 6 Ay", "Son 1 Yıl", "Son 3 Yıl", "Tümü"), index=2)

    end_date = pd.Timestamp.now()
    if time_range == "Son 1 Ay": start_date = end_date - pd.DateOffset(months=1)
    elif time_range == "Son 6 Ay": start_date = end_date - pd.DateOffset(months=6)
    elif time_range == "Son 1 Yıl": start_date = end_date - pd.DateOffset(years=1)
    elif time_range == "Son 3 Yıl": start_date = end_date - pd.DateOffset(years=3)
    else: start_date = pd.Timestamp('1990-01-01')

    st.sidebar.header("Gösterge Ayarları")
    show_sma = st.sidebar.checkbox("Hareketli Ortalamalar (SMA)", value=True)
    show_rsi = st.sidebar.checkbox("RSI Göstergesi", value=True)

    st.sidebar.header("Yapay Zeka")
    run_ai_prediction = st.sidebar.button("AI ile Sinyal Üret")

    try:
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

        if data.empty:
            st.error(f"'{ticker_symbol}' için veri bulunamadı.")
        else:
            st.success(f"'{ticker_symbol}' verileri {start_date.strftime('%Y-%m-%d')} ve {end_date.strftime('%Y-%m-%d')} aralığında başarıyla çekildi.")
            st.subheader(f"{ticker_symbol} Fiyat Grafiği ve Teknik Göstergeler")
            fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Fiyat')])
            
            if show_sma:
                data['SMA20'] = data['Close'].rolling(window=20).mean()
                data['SMA50'] = data['Close'].rolling(window=50).mean()
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], mode='lines', name='SMA20'))
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], mode='lines', name='SMA50'))

            if run_ai_prediction:
                with st.spinner('Yapay zeka modeli eğitiliyor...'):
                    from sklearn.preprocessing import MinMaxScaler
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import LSTM, Dense
                    
                    if len(data) < 61:
                        st.warning(f"AI modeli için yeterli veri yok (en az 61 gün gerekli).")
                    else:
                        close_data = data['Close'].values.reshape(-1, 1)
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        scaled_data = scaler.fit_transform(close_data)
                        X_train, y_train = [], []
                        for i in range(60, len(scaled_data)):
                            X_train.append(scaled_data[i-60:i, 0])
                            y_train.append(scaled_data[i, 0])
                        X_train, y_train = np.array(X_train), np.array(y_train)
                        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                        model = Sequential([ LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)), LSTM(50, return_sequences=False), Dense(25), Dense(1) ])
                        model.compile(optimizer='adam', loss='mean_squared_error')
                        history = model.fit(X_train, y_train, batch_size=1, epochs=10, verbose=0)
                        train_predictions = scaler.inverse_transform(model.predict(X_train))
                        prediction_data = data[60:].copy()
                        prediction_data['Predictions'] = train_predictions
                        fig.add_trace(go.Scatter(x=prediction_data.index, y=prediction_data['Predictions'], mode='lines', name='AI Tahminleri'))
                        last_60_days = scaled_data[-60:]
                        X_test = np.reshape(last_60_days, (1, 60, 1))
                        predicted_price = scaler.inverse_transform(model.predict(X_test))[0][0]
                        st.success(f"Yapay Zeka Modeli Başarıyla Çalıştırıldı!")
                        st.subheader("Yapay Zeka Değerlendirmesi")
                        final_loss = history.history['loss'][-1]
                        confidence_score = max(0, 100 * (1 - final_loss * 100))
                        if confidence_score < 50: confidence_text = "Düşük"
                        elif 50 <= confidence_score < 75: confidence_text = "Orta"
                        else: confidence_text = "Yüksek"
                        last_price = data['Close'].iloc[-1]
                        percentage_diff = ((predicted_price - last_price) / last_price) * 100
                        signal = "BEKLE"
                        if percentage_diff > 5: signal = "GÜÇLÜ AL"
                        elif 2 < percentage_diff <= 5: signal = "AL"
                        elif percentage_diff < -5: signal = "GÜÇLÜ SAT"
                        elif -5 <= percentage_diff < -2: signal = "SAT"
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Sinyal", signal, f"{percentage_diff:.2f}%")
                        col2.metric("Güven Puanı", f"{confidence_score:.2f}%", confidence_text)
                        col3.metric("AI Tahmini", f"{predicted_price:.2f} TL", f"Son Fiyat: {last_price:.2f} TL")
            
            fig.update_layout(title=f'{ticker_symbol} Fiyat Grafiği', yaxis_title='Fiyat (TL)', xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

            if show_rsi:
                st.subheader("RSI (Göreceli Güç Endeksi)")
                data.ta.rsi(append=True)
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(x=data.index, y=data['RSI_14'], mode='lines', name='RSI'))
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Aşırı Alım")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Aşırı Satım")
                rsi_fig.update_layout(title=f'{ticker_symbol} RSI Grafiği', yaxis_title='RSI Değeri', yaxis=dict(range=[0, 100]))
                st.plotly_chart(rsi_fig, use_container_width=True)

            st.subheader("Hacim Analizi")
            st.bar_chart(data['Volume'])

            st.subheader("Son Güncel Veriler")
            st.dataframe(data.tail(10))

    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")
