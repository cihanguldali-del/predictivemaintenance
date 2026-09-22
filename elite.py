import pandas as pd
import numpy as np
import os

# ================= MÜHENDİSLİK AYARLARI (GÖRECELİ YOLLAR) =================
# Kodun çalıştığı klasörü 'merkez' kabul eder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ham verilerin olduğu yer (Flaş bellekte kodun yanındaki klasör)
DATA_DIR = BASE_DIR

# Çıktıların yazılacağı yer
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "model_ready")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ŞAMPİYON KADRO (Özellik Listesi)
SELECTED_FEATURES = [
    'wav_E4_H', 'wav_E7_V', 'wav_E1_H', 'wav_E3_H', 'wav_E5_H',
    'wav_E2_H', 'wav_E4_V',
    'kurt_H', 'peak_H', 'rms_H', 'var_H', 'impulse_H', 'crest_H',
    'freq_var_H', 'mf_H',
    'temperature', 'condition'
]

TARGET_COLS = ['rul_seconds', 'bearing', 'subset']

def smooth_features(df, features, alpha=0.3):
    """Veriyi yumuşatır (Spike'ları temizler)"""
    df_smoothed = df.copy()
    bearings = df['bearing'].unique()
    print(f"   -> Özellikler yumuşatılıyor (Alpha: {alpha})...")

    for b in bearings:
        mask = df['bearing'] == b
        for col in features:
            if col != 'condition' and col in df_smoothed.columns:
                df_smoothed.loc[mask, col] = df.loc[mask, col].ewm(alpha=alpha, adjust=False).mean()

    return df_smoothed

def process_and_save(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"⚠️ Uyarı: {filename} bulunamadı! Yol: {path}")
        return

    print(f">>> İşleniyor: {filename}")
    df = pd.read_csv(path)

    available_feats = [f for f in SELECTED_FEATURES if f in df.columns]
    final_cols = available_feats + TARGET_COLS

    df_elite = df[final_cols].copy()

    # Temel yumuşatma (Mühendislik filtresi)
    df_elite = smooth_features(df_elite, available_feats, alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "ELITE_" + filename)
    df_elite.to_csv(save_path, index=False)
    print(f"    [KAYDEDİLDİ] {save_path}")

def main():
    print(f"--- TAŞINABİLİR ELITE SETİ OLUŞTURUCU (Üs: {BASE_DIR}) ---")
    process_and_save("learning_dataset_final.csv")
    process_and_save("test_dataset_final.csv")
    print("✅ ADIM 1 TAMAM.")

if __name__ == "__main__":
    main()