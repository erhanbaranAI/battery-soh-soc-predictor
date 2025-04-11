# 3-train_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# === Ayarlar ===
DATA_PATH = Path("model-2/processed/all_batteries_processed.csv")
SAVE_PATH = Path("model-2/models")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# === Veriyi Yükle ===
df = pd.read_csv(DATA_PATH)

# === Batarya Tipini Label Encode Et ===
le = LabelEncoder()
df["battery"] = le.fit_transform(df["battery"])

# === Girdi ve hedef değişkenleri ayarla ===
FEATURES = ["cycle", "t_mean_IC5", "battery"]
TARGETS = ["discharge_duration", "soc"]

# === Özellik sırasını kaydet (API için)
joblib.dump(FEATURES, SAVE_PATH / "features_order.pkl")
print("✅ Özellik sırası kaydedildi: features_order.pkl")

# === Model Eğitimi ve Değerlendirme ===
def evaluate_model(y_true, y_pred, title):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n📊 {title} Sonuçları")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")

    # Grafik kaydet
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.xlabel("Gerçek")
    plt.ylabel("Tahmin")
    plt.title(f"{title} Tahmin vs Gerçek")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.tight_layout()
    plt.savefig(SAVE_PATH / f"{title}_plot.png")
    plt.close()

# === Model eğitimi başlasın ===
for target in TARGETS:
    X = df[FEATURES]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    print(f"\n🚀 Eğitiliyor: RandomForest ({target})")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    evaluate_model(y_test, preds, title=f"RandomForest_{target}")

    # Kaydet
    model_filename = SAVE_PATH / f"RandomForest_{target}.pkl"
    joblib.dump(model, model_filename)
    print(f"💾 Model kaydedildi: {model_filename}")

print("\n✅ Tüm RandomForest modelleri başarıyla eğitildi ve kaydedildi.")
