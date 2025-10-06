StockPriceAI

Hisse senedi fiyat tahmini ve analizine yönelik uçtan uca bir makine öğrenimi projesi. Zaman serisi verilerini indirir, temizler, özellik mühendisliği uygular, çeşitli modelleri (örn. LSTM/GRU) eğitir ve performanslarını karşılaştırır. Son aşamada tek komutla tahmin üretip görselleştirme sağlar.

Not: Proje yapısı kökteki StockPriceAI/ klasöründe organize edilmiştir. GitHub

Özellikler

📥 Veri alma: Seçilen semboller için tarihsel fiyat verileri (örn. Yahoo Finance).

🧹 Ön işleme: Eksik değer yönetimi, ölçekleme, pencereleme (sliding window).

🧠 Modeller: LSTM / GRU tabanlı derin öğrenme ve kıyas için ML tabanlı baseline’lar.

📊 Değerlendirme: MAE/MSE/RMSE, görselleştirme (train/val loss, gerçek vs. tahmin).

🚀 Tahmin: Son modeli yükleyip tek komutla ileriye dönük tahmin üretme.

🖥️ (Opsiyonel) Arayüz: Streamlit ile hızlı demo.

Proje Yapısı (Önerilen)
StockPriceAI/
├─ data/
│  ├─ raw/              # İndirilen ham veriler
│  └─ processed/        # Temizlenmiş/veri setleri
├─ notebooks/           # Keşif ve deney defterleri (EDA, prototipler)
├─ src/
│  ├─ config.py         # Ayarlar ve sabitler
│  ├─ data.py           # Veri indirme/okuma/ön işleme
│  ├─ features.py       # Özellik mühendisliği
│  ├─ models/
│  │  ├─ lstm.py        # LSTM mimarisi
│  │  ├─ gru.py         # GRU mimarisi
│  │  └─ baseline.py    # Baseline modeller
│  ├─ train.py          # Eğitim döngüsü
│  ├─ evaluate.py       # Metrikler ve rapor
│  └─ predict.py        # Kaydedilmiş model ile tahmin
├─ requirements.txt
├─ README.md
└─ app.py               # (Opsiyonel) Streamlit demo


Klasör isimleri ve dosyalar mevcut yapına göre değiştirilebilir. Kökte Python kullanıldığı tespit edilmiştir.

Kurulum
# 1) Ortam
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2) Bağımlılıklar
pip install -r requirements.txt


Eğer requirements.txt yoksa örnek:
pandas numpy scikit-learn matplotlib yfinance tensorflow (veya PyTorch tercih ediyorsan torch torchvision)

Hızlı Başlangıç
1) Veri İndir & Hazırla
python -m src.data --symbol AAPL --start 2015-01-01 --end 2025-01-01 --interval 1d
python -m src.features --symbol AAPL --window 60

2) Eğitim
python -m src.train --model lstm --symbol AAPL --epochs 20 --batch 64
# Alternatif:
python -m src.train --model gru --symbol AAPL --epochs 20 --batch 64

3) Değerlendirme
python -m src.evaluate --symbol AAPL --model_path artifacts/AAPL_lstm_best.pt

4) Tahmin
python -m src.predict --symbol AAPL --model_path artifacts/AAPL_lstm_best.pt --horizon 30

5) (Opsiyonel) Streamlit Demo
streamlit run app.py

Yapılandırma

src/config.py içinde ortak parametreleri yönetebilirsin:

SEQ_LEN (pencere boyutu), TRAIN_SPLIT, VAL_SPLIT

LR, EPOCHS, BATCH_SIZE

Model türü: lstm / gru / baseline

Metrikler & Raporlama

MSE / RMSE / MAE

Öğrenme eğrileri (train/val loss)

Gerçek vs Tahmin grafikleri

Kayıtlar artifacts/ ve/veya reports/ altında saklanır.

Veri Notları

Kaynak: Yahoo Finance / benzeri halka açık veri servisleri.

Uyarı: Finansal veriler gürültülüdür; geçmiş performans geleceği garanti etmez.

Zaman bölgesi ve split/ölçekleme sızıntısına (data leakage) dikkat edin.

Yol Haritası

 Hiperparametre araması (Optuna/Ray Tune)

 Çoklu sembol eğitimi (multi-ticker)

 Özellik seti genişletme (teknik indikatörler: RSI, MACD, Bollinger)

 Model karşılaştırma panosu

 Model versiyonlama (MLflow, DVC)

Katkı

Katkılar memnuniyetle karşılanır! Lütfen:

Fork → branch aç

Açıklayıcı commit mesajları

PR oluştur (test ve örnek eklemeye çalış)

Lisans

Bu proje MIT lisansı ile yayınlanabilir (isteğe bağlı). LICENSE dosyası eklenebilir.

İletişim

Soru/öneri: GitHub Issues

Proje sahibi: beytullahatasoy
