# Ana Sayfa - app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Sayfa ayarlarını yapıyoruz
st.set_page_config(
    page_title="StockPriceAI - Ana Sayfa",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Piyasa Özet Paneli Fonksiyonu ---
@st.cache_data(ttl=600) # Veriyi 10 dakika önbellekte tut
def get_market_data():
    """
    Önemli piyasa verilerini çeken fonksiyon.
    """
    tickers = {
        "BIST 100": "XU100.IS", 
        "USD/TRY": "TRY=X",
        "EUR/TRY": "EURTRY=X",
        "Altın (Ons)": "GC=F" 
    }
    data = {}

    for name, ticker in tickers.items():
        try:
            # Tüm veriler için son 2 günü çekmek yeterli
            hist = yf.Ticker(ticker).history(period="2d", interval="1d")

            if not hist.empty and len(hist) > 1:
                previous_close = hist['Close'].iloc[-2]
                current_price = hist['Close'].iloc[-1]
                change = current_price - previous_close
                percent_change = (change / previous_close) * 100
                data[name] = {
                    "price": f"{current_price:,.2f}",
                    "change": f"{percent_change:+.2f}%"
                }
            else:
                 data[name] = {"price": "Veri Yok", "change": ""}
        except Exception as e:
            st.toast(f"{name} verisi alınamadı.", icon="⚠️")
            data[name] = {"price": "Hata", "change": ""}
            
    return data

# --- Ana Sayfa Arayüzü ---

# 1. Karşılama Mesajı ve Uygulama Tanıtımı (Ortalandı)
st.markdown("<h1 style='text-align: center;'>📈 StockPriceAI'a Hoş Geldiniz</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Yapay zeka destekli finansal analiz platformu</p>", unsafe_allow_html=True)
st.markdown("---") 

# 2. Piyasa Özet Panelini Göster
market_data = get_market_data()

cols = st.columns(4) 
with cols[0]:
    st.metric(label="BIST 100", value=market_data["BIST 100"]["price"], delta=market_data["BIST 100"]["change"])

with cols[1]:
    st.metric(label="USD/TRY", value=market_data["USD/TRY"]["price"], delta=market_data["USD/TRY"]["change"])
with cols[2]:
    st.metric(label="EUR/TRY", value=market_data["EUR/TRY"]["price"], delta=market_data["EUR/TRY"]["change"])
with cols[3]:
    st.metric(label="Altın (Ons Fiyatı)", value=market_data["Altın (Ons)"]["price"], delta=market_data["Altın (Ons)"]["change"])

st.markdown("---")

# 3. Açıklama Bölümü
st.subheader("Neler Yapabilirsiniz?")
st.markdown("""
- **🤖 Model Eğitimi:** Kendi yapay zeka modellerinizi farklı parametrelerle eğitin ve performanslarını takip edin.
- **🔮 AI Tahminleri:** Eğittiğiniz veya hazır modelleri kullanarak geleceğe yönelik fiyat tahminleri ve trend analizleri alın.
- **📊 Detaylı Analiz:** Yakında eklenecek olan BIST100, Kripto ve Global Piyasalar sayfalarında derinlemesine analizler yapın.
""")
st.write("Başlamak için sol taraftaki menüden bir sayfa seçin!")


# "Afilli" Uyarı Metni
st.markdown("""
<style>
.fancy-info-box {
    background-color: #262730;
    border-left: 5px solid #ff4b4b;
    padding: 15px;
    border-radius: 8px;
    margin-top: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    display: flex;
    align-items: center;
}
.fancy-info-box p {
    margin: 0;
    padding-left: 15px;
    color: #FAFAFA;
}
</style>
<div class="fancy-info-box">
    <span>⚠️</span>
    <p>Bu uygulama bir staj projesi kapsamında geliştirilmiştir ve yatırım tavsiyesi niteliği taşımaz.</p>
</div>
""", unsafe_allow_html=True)

