# StockPriceAI

Hisse senedi fiyat tahmini ve analizine yönelik uçtan uca bir makine öğrenimi projesi. Zaman serisi verilerini indirir, temizler, özellik mühendisliği uygular, çeşitli modelleri (örn. LSTM/GRU) eğitir ve performanslarını karşılaştırır. Son aşamada tek komutla tahmin üretip görselleştirme sağlar.

> Not: Proje yapısı kökteki `StockPriceAI/` klasöründe organize edilmiştir.

## Özellikler

- 📥 **Veri alma**: Seçilen semboller için tarihsel fiyat verileri (örn. Yahoo Finance).
- 🧹 **Ön işleme**: Eksik değer yönetimi, ölçekleme, pencereleme (sliding window).
- 🧠 **Modeller**: LSTM / GRU tabanlı derin öğrenme ve ML tabanlı baseline’lar.
- 📊 **Değerlendirme**: MAE/MSE/RMSE, görselleştirme (train/val loss, gerçek vs. tahmin).
- 🚀 **Tahmin**: Son modeli yükleyip ileriye dönük tahmin üretme.
- 🖥️ **(Opsiyonel) Arayüz**: Streamlit ile demo.

## Proje Yapısı 

```text
StockPriceAI/
├─ data/
│  ├─ raw/               # İndirilen ham veriler
│  └─ processed/         # Temizlenmiş/veri setleri
├─ notebooks/            # EDA ve prototipler
├─ src/
│  ├─ config.py          # Ayarlar ve sabitler
│  ├─ data.py            # Veri indirme/okuma/ön işleme
│  ├─ features.py        # Özellik mühendisliği
│  ├─ models/
│  │  ├─ lstm.py         # LSTM mimarisi
│  │  ├─ gru.py          # GRU mimarisi
│  │  └─ baseline.py     # Baseline modeller
│  ├─ train.py           # Eğitim döngüsü
│  ├─ evaluate.py        # Metrikler ve rapor
│  └─ predict.py         # Kaydedilmiş model ile tahmin
├─ requirements.txt
├─ README.md
└─ app.py                # (Opsiyonel) Streamlit demo

Kurulum
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate


Bağımlılıklar:

pip install -r requirements.txt


Örnek requirements.txt (yoksa):

pandas
numpy
scikit-learn
matplotlib
yfinance
tensorflow    # veya torch

Kullanım
1) Veri hazırlama
python -m src.data --symbol AAPL --start 2015-01-01 --end 2025-01-01 --interval 1d
python -m src.features --symbol AAPL --window 60

2) Model eğitimi
python -m src.train --model lstm --symbol AAPL --epochs 20 --batch 64
# Alternatif:
python -m src.train --model gru --symbol AAPL --epochs 20 --batch 64

3) Değerlendirme
python -m src.evaluate --symbol AAPL --model_path artifacts/AAPL_lstm_best.pt

4) Tahmin
python -m src.predict --symbol AAPL --model_path artifacts/AAPL_lstm_best.pt --horizon 30

5) (Opsiyonel) Arayüz
streamlit run app.py

Yapılandırma

src/config.py üzerinden:

SEQ_LEN (pencere boyutu)

TRAIN_SPLIT / VAL_SPLIT

EPOCHS / BATCH_SIZE / LR

Model tipi (lstm / gru / baseline)

Yol Haritası

Hiperparametre araması

Çoklu sembol desteği

Teknik indikatörler (RSI, MACD vb.)

Model karşılaştırmaları

MLflow / DVC ile model takibi

Katkı

Fork alın

Yeni branch açın

Commit → Push

Pull Request gönderin

Lisans

İsteğe bağlı olarak MIT lisansı eklenebilir.

İletişim

Sorular ve öneriler için GitHub Issues bölümünü kullanabilirsiniz.
