# pages/2_🤖_Model_Eğitimi.py

import streamlit as st

# --- Sayfa Ayarları ---
st.set_page_config(layout="wide")
st.title("🤖 Model Eğitimi & Performans Takibi")

# --- Kenar Çubuğu (Sidebar) ---
st.sidebar.header("Eğitim Parametreleri")

# Model Tipi Seçimi
model_tipi = st.sidebar.selectbox(
    "Model Tipi",
    ("LSTM", "GRU", "Transformer"),
    help="Kullanılacak derin öğrenme modelinin mimarisini seçin."
)

st.sidebar.markdown("---")

# Model Hiperparametreleri
st.sidebar.subheader("Model Hiperparametreleri")

hidden_size = st.sidebar.slider(
    "Hidden Size",
    min_value=32,
    max_value=512,
    value=128,
    step=32,
    help="Modeldeki her bir katmanın nöron sayısı. Yüksek değerler modelin kapasitesini artırır ama eğitimi yavaşlatır."
)

layer_sayisi = st.sidebar.slider(
    "Layer Sayısı",
    min_value=1,
    max_value=4,
    value=2,
    help="Modeldeki katman sayısı."
)

dropout = st.sidebar.slider(
    "Dropout Oranı",
    min_value=0.0,
    max_value=0.5,
    value=0.2,
    step=0.05,
    help="Modelin ezberlemesini (overfitting) önlemek için kullanılan bir tekniktir."
)

st.sidebar.markdown("---")

# Eğitim Parametreleri
st.sidebar.subheader("Eğitim Parametreleri")

epoch_sayisi = st.sidebar.slider(
    "Epoch Sayısı",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
    help="Modelin tüm veri setini kaç kez baştan sona inceleyeceği."
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    (16, 32, 64, 128),
    index=1,
    help="Modelin her bir adımda kaç veri örneğini aynı anda işleyeceği."
)


# --- Ana Ekran ---
st.subheader("Model Eğitimi")

col1, col2 = st.columns([3, 2])

with col1:
    hisse_secimi = st.selectbox(
        "Hisse Senedi Seçin:",
        ("AAPL", "MSFT", "GOOGL", "THYAO.IS", "GARAN.IS"),
        help="Modelin eğitileceği hisse senedini seçin."
    )

with col2:
    st.info("Veri seti hazır: 662 kayıt.", icon="ℹ️")


st.info("Veri Hazırlandı.", icon="✅")

if st.button("🔴 Eğitimi Başlat", use_container_width=True):
    with st.spinner("Model, seçilen parametrelerle eğitiliyor..."):
        # Arka plan mantığı bir sonraki adımda buraya eklenecek
        st.success("Eğitim Başarıyla Tamamlandı!")

st.markdown("---")

st.subheader("Performans Analizi")
st.info("Eğitim başlatıldıktan sonra performans grafikleri ve sonuçları burada görünecektir.")