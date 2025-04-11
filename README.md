# 🔋 Lithium-Ion Battery SoH & SoC Tahmin Sistemi

Bu proje, NASA'nın yayımladığı batarya veri setleri kullanılarak, bataryaların **Sağlık Durumu (SoH)** ve **Şarj Durumu (SoC)** değerlerini tahmin eden bir makine öğrenmesi modelini içermektedir. Model Flask tabanlı bir REST API ile sunulmakta, Streamlit arayüzü ile kullanıcıya görselleştirme ve etkileşimli kullanım imkânı sunmaktadır.

## 📦 Proje Yapısı

```
ErhanBaran-Case/
├── app/
│   ├── 4-serve_model.py          # Flask REST API
│   ├── 5-app.py                  # Streamlit Arayüz
│   ├── requirements.txt          # Gerekli Python paketleri
│   ├── model-2/
│   │   └── models/               # Eğitilmiş modeller (.pkl)
│   └── data/processed/           # Özetlenmiş veri
├── Dockerfile                    # Flask için Dockerfile
├── Dockerfile2                   # Streamlit için Dockerfile
├── docker-compose.yml            # Tüm sistemi ayağa kaldıran yapı
```

## 🚀 Kurulum ve Çalıştırma

1. Bu repoyu klonlayın:
```bash
git clone https://github.com/kullaniciadi/proje-adi.git
cd proje-adi
```

2. Docker Compose ile uygulamayı başlatın:
```bash
docker compose up --build
```

3. Servisler başlatıldıktan sonra:

- **API (Flask)** → `http://localhost:5000/predict`  
- **Arayüz (Streamlit)** → `http://localhost:8501`

## 📨 API Kullanımı

### Endpoint
```http
POST /predict
```

### Örnek JSON Gönderimi
```json
{
  "features": [310, 0.0, 0]
}
```

### Dönüş
```json
{
  "predicted_soh": 3107.44,
  "predicted_soc": 0.838
}
```

## 🛠 Kullanılan Teknolojiler

- Python (Flask, Streamlit, scikit-learn, pandas, joblib)
- Docker & Docker Compose
- NASA Battery Dataset (B0005, B0006, B0018)

## 👨‍🔬 Geliştirici

Erhan Baran  
📧 lerhanbaranl@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/erhan-baran/)