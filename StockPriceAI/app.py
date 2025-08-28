# 1. Gerekli Kütüphanelerin Yüklenmesi
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# 2. Sayfa ve Arayüz Ayarları
st.set_page_config(page_title="StockPriceAI Prototip", layout="wide")
st.title("📈 StockPriceAI - Hisse Senedi Analiz Platformu")

# 3. Kenar Çubuğu (Sidebar) Ayarları
st.sidebar.header("Analiz Ayarları")
ticker_symbol = st.sidebar.text_input("Hisse Senedi Kodu (Örn: THYAO.IS)", "THYAO.IS").upper()
start_date = st.sidebar.date_input("Başlangıç Tarihi", pd.to_datetime('2023-01-01'))
end_date = st.sidebar.date_input("Bitiş Tarihi", pd.to_datetime('today'))

# 4. Ana Uygulama Bloğu
try:
    # Veriyi yfinance üzerinden çek
    data = yf.download(ticker_symbol, start=start_date, end=end_date)

    # yfinance'den gelen karmaşık sütun isimlerini düzelt (Örn: ('Open', 'THYAO.IS') -> 'Open')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Veri setinin boş olup olmadığını kontrol et
    if data.empty:
        st.error(f"'{ticker_symbol}' için veri bulunamadı. Lütfen hisse senedi kodunu kontrol edin (BIST için sonuna '.IS' eklemeyi unutmayın).")
    else:
        st.success(f"'{ticker_symbol}' verileri başarıyla çekildi.")

        # --- Fiyat Grafiği ve Teknik Göstergeler ---
        st.subheader(f"{ticker_symbol} Fiyat Grafiği ve Teknik Göstergeler")

        # 20 ve 50 günlük Hareketli Ortalamaları (SMA) hesapla
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()

        # Plotly ile mum grafiğini oluştur
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name='Fiyat')])

        # Hareketli Ortalamaları grafiğe ekle
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], mode='lines', name='20-Günlük Ortalama (SMA)'))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], mode='lines', name='50-Günlük Ortalama (SMA)'))

        # Grafik ayarlarını güncelle
        fig.update_layout(
            title=f'{ticker_symbol} Fiyat ve Hareketli Ortalamalar',
            yaxis_title='Fiyat (TL)',
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Hacim Analizi ---
        st.subheader("Hacim Analizi")
        st.bar_chart(data['Volume'])

        # --- Son Veriler Tablosu ---
        st.subheader("Son Güncel Veriler")
        st.dataframe(data.tail(10))

except Exception as e:
    st.error(f"Bir hata oluştu: {e}")