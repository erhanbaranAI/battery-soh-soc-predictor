import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

st.set_page_config(page_title="Battery SoH & SoC Prediction", page_icon="🔋", layout="wide")

# === BATARYA CYCLE SINIRLARI ===
min_cycles = {"B0005": 2, "B0006": 2, "B0018": 3}
max_cycles = {"B0005": 614, "B0006": 614, "B0018": 319}

# === Isıl Özellik Aralığı (t_mean_IC5) ===
t_min = -1.6
t_max = 1.6

# === Arayüz Sekmeleri ===
tab1, tab2 = st.tabs(["🔋 Tahmin Aracı", "📊 Genel Analiz"])

# === TAHMİN ARACI TAB ===
with tab1:
    with st.sidebar:
        st.markdown("### 🔧 Girdi Değerleri")

        battery_type = st.radio("🔋 Batarya Tipi", ["B0005", "B0006", "B0018"])
        battery_encoded = {"B0005": 0, "B0006": 1, "B0018": 2}[battery_type]

        cycle = st.slider("🔄 Döngü Sayısı (Cycle)", min_cycles[battery_type], max_cycles[battery_type], step=1)

        st.markdown("### 🌡️ Isıl Özellik (t_mean_IC5)")
        t_mean_ic5 = st.slider("Slider ile seçin", t_min, t_max, value=0.0, step=0.01)

    st.markdown("## 🔋 Lithium-Ion Battery SoH & SoC Prediction")
    st.markdown("Batarya verilerinize göre **deşarj süresi (SoH)** ve **şarj durumu (SoC)** tahmin edin.")

    if st.button("🔍 Tahmin Et"):
        input_data = {
            "cycle": cycle,
            "t_mean_IC5": t_mean_ic5,
            "battery": battery_encoded
        }

        try: 
            response = requests.post("http://serve-model:5000/predict", json={"features": list(input_data.values())})

            if response.status_code == 200:
                prediction = response.json()
                soh = prediction.get("predicted_soh")
                soc = prediction.get("predicted_soc")

                st.session_state["predicted_soh"] = soh
                st.session_state["last_cycle"] = cycle
                st.session_state["battery_type"] = battery_type

                col1, col2 = st.columns(2)
                col1.subheader("🔋 SoH (Deşarj Süresi)")
                col1.metric(label="Saniye", value=f"{soh:.2f}")

                col2.subheader("⚡ SoC (Şarj Durumu)")
                col2.metric(label="Yüzde", value=f"{soc * 100:.2f} %")

                # ✅ GERÇEK VERİLERİ YÜKLE
                df_all = pd.read_csv("model-2/processed/all_batteries_processed.csv")
                df_all_battery = df_all[df_all["battery"] == battery_encoded]
                t_ic5_real = df_all_battery.loc[df_all_battery["cycle"] == cycle, "t_mean_IC5"].values

                if len(t_ic5_real) > 0:
                    with st.expander("🔍 Gerçek Değerler (Seçilen Cycle)"):
                        st.markdown(f"""
                        - 🔄 **Cycle:** {cycle}  
                        - 🌡️ **Gerçek t_mean_IC5:** {t_ic5_real[0]:.4f}  
                        - 🔋 **Battery:** {battery_type}
                        """)

                # === GRAFİKLER İÇİN ÖZET DOSYASINI KULLAN
                selected_battery = battery_type
                path = f"data/cycle_summaries/{selected_battery}_summary.csv"
                df = pd.read_csv(path)

                cycle_original = cycle
                soh_predicted = soh

                df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
                df.dropna(subset=["cycle"], inplace=True)
                df["cycle"] = df["cycle"].astype(int)

                # === SoH Görselleştirme ===
                soh_values = df.groupby("cycle")["discharge_duration"].mean().reset_index()
                real_soh_at_predicted_cycle = soh_values.loc[soh_values["cycle"] == cycle_original, "discharge_duration"].values

                st.markdown("### 📈 SoH Tahmini ve Gerçek Değer Görselleştirmesi")
                plt.figure(figsize=(10, 4))
                plt.plot(soh_values["cycle"], soh_values["discharge_duration"], linestyle='dotted', label="Gerçek SoH Trend", color="blue")
                plt.scatter(cycle_original, soh_predicted, color="red", s=100, label="Tahmin Edilen SoH")
                plt.text(cycle_original, soh_predicted + 40, f"y: {soh_predicted:.0f}", color="red", fontsize=9)

                if len(real_soh_at_predicted_cycle) > 0:
                    actual_soh = real_soh_at_predicted_cycle[0]
                    error = abs(actual_soh - soh_predicted)
                    accuracy_percent = 100 - ((error / actual_soh) * 100)

                    st.markdown(f"🧮 **Tahmin Hata Payı (Absolute Error)**: `{error:.2f} saniye`")
                    st.markdown(f"🎯 **Doğruluk (Accuracy)**: `{accuracy_percent:.2f} %`")

                    plt.scatter(cycle_original, actual_soh, color="black", s=100, label="Gerçek SoH")
                    plt.text(cycle_original + 1, actual_soh + 40, f"y: {actual_soh:.0f}", color="black", fontsize=9)

                plt.xlabel("Cycle")
                plt.ylabel("Discharge Duration (SoH)")
                plt.title(f"SoH Trend - Battery {selected_battery}")
                plt.legend()
                st.pyplot(plt.gcf())

                # === SoC Görselleştirme ===
                real_soc_at_predicted_cycle = df.loc[df["cycle"] == cycle_original, "soc"].values
                st.markdown("### ⚡ SoC Tahmini ve Gerçek Değer Görselleştirmesi")

                plt.figure(figsize=(10, 4))
                plt.plot(df["cycle"], df["soc"], linestyle='dotted', label="Gerçek SoC Trend", color="green")
                plt.scatter(cycle_original, soc, color="orange", s=100, label="Tahmin Edilen SoC")
                plt.text(cycle_original, soc + 0.01, f"y: {soc * 100:.1f} %", color="orange", fontsize=9)

                if len(real_soc_at_predicted_cycle) > 0:
                    actual_soc = real_soc_at_predicted_cycle[0]
                    soc_error = abs(actual_soc - soc)
                    soc_accuracy = 100 - (soc_error * 100)

                    st.markdown(f"📉 **SoC Tahmin Hata Payı**: `{soc_error:.4f}` (`±{soc_error * 100:.2f} %`)")
                    st.markdown(f"🎯 **SoC Doğruluk**: `{soc_accuracy:.2f} %`")

                    plt.scatter(cycle_original, actual_soc, color="black", s=100, label="Gerçek SoC")
                    plt.text(cycle_original + 1, actual_soc + 0.01, f"y: {actual_soc * 100:.1f} %", color="black", fontsize=9)

                plt.xlabel("Cycle")
                plt.ylabel("State of Charge (SoC)")
                plt.title(f"SoC Trend - Battery {selected_battery}")
                plt.legend()
                st.pyplot(plt.gcf())

            else:
                st.error("❌ API'den cevap alınamadı.")
        except Exception as e:
            st.error(f"❌ Hata oluştu: {str(e)}")

# === GENEL ANALİZ ===
with tab2:
    st.subheader("📊 Tüm Bataryalar İçin Isıl Özellik Dağılımı")
    df_all = pd.read_csv("model-2/processed/all_batteries_processed.csv")

    plt.figure(figsize=(10, 5))
    df_all["battery"] = df_all["battery"].map({0: "B0005", 1: "B0006", 2: "B0018"})
    df_all = df_all[df_all["t_mean_IC5"].between(t_min, t_max)]
    for bat in df_all["battery"].unique():
        subset = df_all[df_all["battery"] == bat]
        plt.hist(subset["t_mean_IC5"], bins=40, alpha=0.5, label=bat)
    plt.xlabel("t_mean_IC5 (Isıl Özellik)")
    plt.ylabel("Adet")
    plt.legend()
    plt.title("Isıl Özellik Dağılımı")
    st.pyplot(plt.gcf())
