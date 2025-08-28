# Ana Sayfa - app.py

import streamlit as st
import yfinance as yf
import pandas as pd

# Sayfa ayarlarını yapıyoruz
st.set_page_config(
    page_title="StockPriceAI - Ana Sayfa",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Piyasa Özet Paneli Fonksiyonu ---
def get_market_data():
    """
    Önemli piyasa verilerini çeken fonksiyon.
    """
    tickers = {
        "BIST 100": "^XU100",
        "USD/TRY": "TRY=X",
        "EUR/TRY": "EURTRY=X",
        "Altın (Gram)": "GC=F" 
    }
    data = {}
    for name, ticker in tickers.items():
        # yfinance'ten son 2 günlük veriyi çekiyoruz ki dünkü kapanışı bulabilelim
        hist = yf.Ticker(ticker).history(period="2d")
        if not hist.empty:
            previous_close = hist['Close'].iloc[0]
            current_price = hist['Close'].iloc[1]
            change = current_price - previous_close
            percent_change = (change / previous_close) * 100
            data[name] = {
                "price": f"{current_price:,.2f}",
                "change": f"{percent_change:+.2f}%"
            }
        else:
            data[name] = {"price": "Veri Alınamadı", "change": ""}
            
    return data

# --- Ana Sayfa Arayüzü ---

# 1. Piyasa Özet Panelini Göster
market_data = get_market_data()
st.subheader("Güncel Piyasa Verileri")

cols = st.columns(4) # 4 sütunlu bir yapı oluşturuyoruz
with cols[0]:
    st.metric(label="BIST 100", value=market_data["BIST 100"]["price"], delta=market_data["BIST 100"]["change"])
with cols[1]:
    st.metric(label="USD/TRY", value=market_data["USD/TRY"]["price"], delta=market_data["USD/TRY"]["change"])
with cols[2]:
    st.metric(label="EUR/TRY", value=market_data["EUR/TRY"]["price"], delta=market_data["EUR/TRY"]["change"])
with cols[3]:
    st.metric(label="Altın (Ons Fiyatı)", value=market_data["Altın (Gram)"]["price"], delta=market_data["Altın (Gram)"]["change"])

st.markdown("---") # Ayırıcı çizgi

# 2. Karşılama Mesajı ve Uygulama Tanıtımı
st.title("📈 StockPriceAI'a Hoş Geldiniz")

st.markdown("""
StockPriceAI, BIST, kripto ve global piyasalardaki varlıkları analiz etmenize olanak tanıyan, 
yapay zeka destekli bir finansal analiz platformudur. 

**Neler Yapabilirsiniz?**

- **Detaylı Analiz:** Kenar çubuğundaki menüden ilgilendiğiniz piyasayı seçerek başlayın.
- **Teknik Göstergeler:** Hareketli ortalamalar (SMA) ve RSI gibi popüler göstergeleri grafikler üzerinde inceleyin.
- **AI Destekli Sinyaller:** Gelişmiş LSTM sinir ağı modelimizi kullanarak gelecek gün için "AL/SAT/BEKLE" tavsiyeleri ve güven puanları alın.

Başlamak için sol taraftaki menüden bir sayfa seçin!
""")

st.info("Bu uygulama bir staj projesi kapsamında geliştirilmiştir ve yatırım tavsiyesi niteliği taşımaz.", icon="⚠️")