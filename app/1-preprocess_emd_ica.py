# 1-preprocess_emd_ica.py

import os
import pandas as pd
import numpy as np

from PyEMD import EMD
from sklearn.decomposition import FastICA
from pathlib import Path

# === Yapılandırmalar ===
BATTERIES = ["B0005", "B0006", "B0018"]
DATA_PATH = Path("data/cycle_summaries")
SAVE_PATH = Path("model-2/processed")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# === Ayarlanabilir Parametreler ===
IMF_COUNT = 5  # Her sinyalden çıkarılacak IMF sayısı
IC_COUNT = 5   # ICA ile çıkarılacak bileşen sayısı

# === İşlenecek sinyaller (sütunlar) ===
SIGNALS = ["v_mean", "i_mean", "t_mean"]

# === İşlem Başlat ===
all_features = []

def extract_emd_ica_features(df: pd.DataFrame, battery_name: str):
    features = {"cycle": df["cycle"].values}
    
    for signal in SIGNALS:
        signal_data = df[signal].values
        emd = EMD()
        imfs = emd(signal_data)

        if imfs.shape[0] < IMF_COUNT:
            padding = IMF_COUNT - imfs.shape[0]
            imfs = np.vstack([imfs, np.zeros((padding, imfs.shape[1]))])

        imfs = imfs[:IMF_COUNT, :].T  # shape: (n_samples, IMF_COUNT)

        # ICA Uygula
        ica = FastICA(n_components=IC_COUNT, random_state=42)
        try:
            ics = ica.fit_transform(imfs)
        except ValueError:
            ics = np.zeros_like(imfs[:, :IC_COUNT])

        for i in range(ics.shape[1]):
            features[f"{signal}_IC{i+1}"] = ics[:, i]

    # Hedef sütunları ekle
    features["discharge_duration"] = df["discharge_duration"].values
    features["soc"] = df["soc"].values
    features["battery"] = battery_name

    return pd.DataFrame(features)


for battery in BATTERIES:
    file_path = DATA_PATH / f"{battery}_summary.csv"
    df = pd.read_csv(file_path)

    print(f"\n🔍 İşleniyor: {file_path}")
    processed = extract_emd_ica_features(df, battery)
    processed.to_csv(SAVE_PATH / f"{battery}_processed.csv", index=False)
    all_features.append(processed)

# Hepsini birleştir
full_df = pd.concat(all_features, ignore_index=True)
full_df.to_csv(SAVE_PATH / "all_batteries_processed.csv", index=False)
print("\n✅ Tüm batarya verileri başarıyla işlendi ve kaydedildi.")
