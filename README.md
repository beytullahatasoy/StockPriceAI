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
│  ├─ raw/
│  └─ processed/
├─ notebooks/
├─ src/
│  ├─ config.py
│  ├─ data.py
│  ├─ features.py
│  ├─ models/
│  │  ├─ lstm.py
│  │  ├─ gru.py
│  │  └─ baseline.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ predict.py
├─ requirements.txt
├─ README.md
└─ app.py
```

## Kurulum

### Sanal ortam

```bash
python -m venv .venv
```

### Ortamı aktifleştirme

Windows:
```bash
.venv\Scripts\activate
```

macOS / Linux:
```bash
source .venv/bin/activate
```

### Bağımlılıklar

```bash
pip install -r requirements.txt
```

`requirements.txt` yoksa örnek:

```text
pandas
numpy
scikit-learn
matplotlib
yfinance
tensorflow   # veya torch
```

## Kullanım

### 1) Veri hazırlama

```bash
python -m src.data --symbol AAPL --start 2015-01-01 --end 2025-01-01 --interval 1d
python -m src.features --symbol AAPL --window 60
```

### 2) Model eğitimi

```bash
python -m src.train --model lstm --symbol AAPL --epochs 20 --batch 64
# Alternatif:
python -m src.train --model gru --symbol AAPL --epochs 20 --batch 64
```

### 3) Değerlendirme

```bash
python -m src.evaluate --symbol AAPL --model_path artifacts/AAPL_lstm_best.pt
```

### 4) Tahmin

```bash
python -m src.predict --symbol AAPL --model_path artifacts/AAPL_lstm_best.pt --horizon 30
```

### 5) Arayüz

```bash
streamlit run app.py
```

## Yapılandırma

`src/config.py` içinden:

- `SEQ_LEN`
- `TRAIN_SPLIT` / `VAL_SPLIT`
- `EPOCHS` / `BATCH_SIZE` / `LR`
- Model tipi (lstm / gru / baseline)

## Yol Haritası

- Hiperparametre optimizasyonu
- Çoklu sembol desteği
- Teknik indikatör eklemeleri (RSI, MACD vb.)
- Model karşılaştırmaları
- MLflow / DVC entegrasyonu

## Katkı

1. Fork alın  
2. Yeni branch açın  
3. Commit & Push  
4. Pull Request gönderin

## İletişim

Sorular ve öneriler için GitHub Issues bölümünü kullanabilirsiniz.
