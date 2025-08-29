# pages/3_🔮_AI_Tahminleri.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Sayfa Ayarları ---
st.set_page_config(layout="wide")
st.title("🔮 AI Tahminleri & Teknik Analiz")

# --- Kenar Çubuğu (Sidebar) ---
with st.sidebar:
    st.header("Tahmin Ayarları")
    
    hisse_secimi = st.selectbox(
        "Hisse Seçin:",
        ("AAPL", "MSFT", "GOOGL", "THYAO.IS", "GARAN.IS"),
        key="tahmin_hisse_secimi"
    )

    tahmin_gunu = st.slider(
        "Tahmin Günü Sayısı",
        min_value=1,
        max_value=30,
        value=7,
        step=1,
        help="Geleceğe yönelik kaç günlük tahmin görmek istediğinizi seçin."
    )

    model_secimi = st.selectbox(
        "Model Seçimi",
        ("LSTM", "GRU", "Ensemble"),
        key="tahmin_model_secimi"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <style>
        div.stButton > button {
            background-color: #FF4B4B; color: white; font-weight: bold;
            border-radius: 8px; border: none; padding: 10px 0; width: 100%;
        }
        div.stButton > button:hover {
            background-color: #E03C3C; color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    if st.button("📈 Veri Yükle & Tahmin Et"):
        pass

# --- Ana Ekran ---
tab1, tab2, tab3 = st.tabs(["Fiyat Tahminleri", "Teknik Sinyaller", "Model Analizi"])

with tab1:
    st.subheader("Tahmin Özeti")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ortalama Tahmin", "$528.97", "+4.3%")
    col2.metric("Maksimum Fiyat", "$546.91", "+7.8%")
    col3.metric("Minimum Fiyat", "$509.03", "+0.4%")
    col4.metric("Risk Seviyesi", "Orta", "Vol: $14.38")

    st.subheader("Fiyat Tahmin Grafiği")
    
    # --- GRAFİK KODU DÜZELTMESİ ---
    fig = go.Figure()
    
    # Örnek geçmiş veriler
    gecmis_tarihler = pd.to_datetime(['2025-08-10', '2025-08-15', '2025-08-20'])
    gecmis_fiyatlar = [485, 510, 520]
    
    # Örnek tahmin verileri
    tahmin_tarihleri = pd.date_range(start='2025-08-21', periods=tahmin_gunu)
    tahmin_fiyatlar = [518, 515, 522, 525, 523, 528, 530, 532, 535, 538, 540, 538, 542, 545, 543, 546, 548, 550, 547, 549, 552, 555, 553, 556, 558, 560, 557, 559, 562, 565][:tahmin_gunu]
    
    # Örnek güven aralığı bantları
    ust_bant = [p + 10 for p in tahmin_fiyatlar] # Tahminin 10 birim üstü
    alt_bant = [p - 10 for p in tahmin_fiyatlar] # Tahminin 10 birim altı
    
    # Grafiğe çizimleri ekle
    fig.add_trace(go.Scatter(x=gecmis_tarihler, y=gecmis_fiyatlar, mode='lines', name='Geçmiş Fiyatlar', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=tahmin_tarihleri, y=tahmin_fiyatlar, mode='lines', name='AI Tahminleri', line=dict(color='red', dash='dot')))
    
    # Bandı doğru çizmek için ÖNCE ÜST, SONRA ALT bandı ekliyoruz
    fig.add_trace(go.Scatter(x=tahmin_tarihleri, y=ust_bant, fill=None, mode='lines', line_color='rgba(255,0,0,0.1)', showlegend=False))
    fig.add_trace(go.Scatter(x=tahmin_tarihleri, y=alt_bant, fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.1)', name='Güven Aralığı'))
    
    fig.update_layout(title=f"{hisse_secimi} - AI Fiyat Tahminleri", yaxis_title="Fiyat")
    st.plotly_chart(fig, use_container_width=True)
    # --- DÜZELTME SONU ---

    st.subheader("Trend ve Güven Analizi")
    col_trend1, col_trend2 = st.columns(2)
    with col_trend1:
        st.info("Genel Trend: DÜŞÜŞ")
    with col_trend2:
        st.info("Kısa Vadeli (3 gün): $484.34")
        st.info("Orta Vadeli (7 gün): $482.56")
    
with tab2:
    st.subheader("Teknik Sinyaller")
    st.info("Teknik göstergelere dayalı AL/SAT sinyalleri burada gösterilecektir.")
    
with tab3:
    st.subheader("Model Performans Analizi")
    st.write("**Model Karşılaştırması:**")
    perf_data = {"Model": ["LSTM", "BiLSTM", "LSTM+Attention", "Ensemble"], "Accuracy": [85.2, 87.8, 89.1, 91.3], "MAE": [2.14, 1.98, 1.87, 1.72], "RMSE": [3.21, 2.95, 2.78, 2.54], "Sharpe Ratio": [1.23, 1.34, 1.42, 1.58]}
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df, use_container_width=True)