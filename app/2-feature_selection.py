# 2-feature_selection.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# === Dosya Yolları ===
DATA_PATH = Path("model-2/processed/all_batteries_processed.csv")
SAVE_PATH = Path("model-2/feature_selection")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# === Veriyi Yükle ===
df = pd.read_csv(DATA_PATH)

# === Kategorik Değişkenleri Kaldır ===
df_numeric = df.drop(columns=["battery"])

# === Korelasyon Matrisi ===
correlation_matrix = df_numeric.corr()

# === Korelasyon Isı Haritası ===
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", center=0)
plt.title("🔍 Korelasyon Matrisi (Heatmap)", fontsize=14)
plt.tight_layout()
plt.savefig(SAVE_PATH / "correlation_heatmap.png")
plt.close()

# === Hedef Değişkenle En İlgili Özellikler ===
target_soh = "discharge_duration"
target_soc = "soc"

def get_top_correlated_features(target_column, threshold=0.6):
    correlations = correlation_matrix[target_column].drop(target_column)
    top_features = correlations[correlations.abs() >= threshold].sort_values(key=abs, ascending=False)
    return top_features

top_soh = get_top_correlated_features(target_soh)
top_soc = get_top_correlated_features(target_soc)

# === Sonuçları Yazdır ===
print("\n🔋 En Yüksek Korelasyona Sahip Özellikler (SoH):")
print(top_soh)

print("\n⚡ En Yüksek Korelasyona Sahip Özellikler (SoC):")
print(top_soc)

# === Kaydet ===
top_soh.to_csv(SAVE_PATH / "top_features_soh.csv")
top_soc.to_csv(SAVE_PATH / "top_features_soc.csv")
