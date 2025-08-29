# pages/2_🤖_Model_Eğitimi.py

import streamlit as st

# --- Sayfa Ayarları ---
st.set_page_config(layout="wide")
st.title("🤖 Model Eğitimi & Performans Takibi")

# --- Kenar Çubuğu (Sidebar) ---
with st.sidebar:
    st.header("Eğitim Parametreleri")

    # Model Tipi Seçimi
    model_tipi = st.selectbox(
        "Model Tipi",
        ("LSTM", "GRU", "Transformer"),
        help="Kullanılacak derin öğrenme modelinin mimarisini seçin."
    )

    st.markdown("---")

    # Model Hiperparametreleri
    st.subheader("Model Hiperparametreleri")

    hidden_size = st.slider(
        "Hidden Size", min_value=32, max_value=512, value=128, step=32,
        help="Modeldeki her bir katmanın nöron sayısı. Yüksek değerler modelin kapasitesini artırır ama eğitimi yavaşlatır."
    )
    layer_sayisi = st.slider(
        "Layer Sayısı", min_value=1, max_value=4, value=2,
        help="Modeldeki katman sayısı."
    )
    dropout = st.slider(
        "Dropout Oranı", min_value=0.0, max_value=0.5, value=0.2, step=0.05,
        help="Modelin ezberlemesini (overfitting) önlemek için kullanılan bir tekniktir."
    )

    st.markdown("---")

    # Eğitim Parametreleri
    st.subheader("Eğitim Parametreleri")

    epoch_sayisi = st.slider(
        "Epoch Sayısı", min_value=5, max_value=100, value=20, step=5,
        help="Modelin tüm veri setini kaç kez baştan sona inceleyeceği."
    )
    batch_size = st.selectbox(
        "Batch Size", (16, 32, 64, 128), index=1,
        help="Modelin her bir adımda kaç veri örneğini aynı anda işleyeceği."
    )

# --- Ana Ekran ---
# Sekmeli bir yapı oluşturuyoruz
tab1, tab2, tab3 = st.tabs(["Eğitim", "Performans", "Model Bilgileri"])

with tab1:
    st.subheader("Model Eğitimi")
    
    col1, col2 = st.columns([2, 3])
    with col1:
        hisse_secimi = st.selectbox(
            "Hisse Senedi Seçin:",
            ("AAPL", "MSFT", "GOOGL", "THYAO.IS", "GARAN.IS"),
            key="egitim_hisse_secimi"
        )
    with col2:
        # Bu kısım dinamik olacak ama şimdilik sabit
        st.info("MSFT verisi hazır: 662 kayıt.", icon="ℹ️")

    st.info("Veri Hazırlandı.", icon="✅")

    # Buton stili için CSS
    st.markdown("""
    <style>
        div.stButton > button {
            background-color: #FF4B4B;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            padding: 10px 0;
            width: 100%;
        }
        div.stButton > button:hover {
            background-color: #E03C3C;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("🔴 Eğitimi Başlat"):
        # Burası bir sonraki adımda doldurulacak
        with st.spinner("Model eğitiliyor..."):
            pass
        st.success("Eğitim tamamlandı! 'Performans' sekmesini kontrol edin.")
    
    st.markdown("---")
    st.subheader("Demo Eğitim")
    if st.button("▶️ Demo Eğitimi Başlat"):
        # Burası da daha sonra doldurulabilir
        st.info("Demo eğitimi özelliği yakında eklenecektir.")

with tab2:
    st.subheader("Performans Analizi")
    st.info("Eğitim başlatıldıktan sonra performans grafikleri ve sonuçları burada görünecektir.")
    
    # Sonuçların gösterileceği yer için bir yer tutucu
    st.write("**Son Eğitim Sonuçları:**")
    
    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
    col_res1.metric("Model Tipi", "LSTM")
    col_res2.metric("Eğitim Süresi", "10.86s")
    col_res3.metric("En İyi Val Loss", "0.012379")
    col_res4.metric("Toplam Epochs", "12")


with tab3:
    st.subheader("Model Mimarisi Bilgileri")
    
    # Model bilgilerini göstermek için bir yer tutucu
    st.code("""
{
    "model_type": "LSTM",
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "epochs": 20,
    "batch_size": 32,
    "sequence_length": 60,
    "train_split": 0.8
}
    """, language="json")