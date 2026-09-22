import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ================= MÜHENDİSLİK AYARLARI (GÖRECEKLİ YOLLAR) =================
# ================= TAŞINABİLİR AYARLAR =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


DATA_DIR = os.path.join(BASE_DIR, "data", "model_ready_prod_WITH_TEMP")

DEFAULT_RESULTS_DIR = os.path.join(DATA_DIR, "training_results_classic")
TEST_CSV_PATH = os.path.join(BASE_DIR, "data", "model_ready", "ELITE_test_dataset_final.csv")

SEQUENCE_LENGTH = 30
# KADRO
SHOWCASE_BEARINGS = ["Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7", "Bearing3_3"]


def apply_physical_limit(y_pred, bearing_features, threshold=20.0):
    """
    Fiziksel Emniyet Sınırı: Eğer peak ivme 20g'yi geçtiyse,
    PRONOSTIA standartlarına göre rulman bitmiştir (RUL=0).
    """
    # 'peak_H' özellik 20g
    if 'peak_H' in bearing_features.columns:
        # Pencerelenmiş veriye uyması için özelliklerin son SEQUENCE_LENGTH-1 satırını kesiyoruz
        vibration_vals = bearing_features['peak_H'].values[SEQUENCE_LENGTH - 1:]

        # 20g'nin aşıldığı ilk anı bul
        limit_indices = np.where(vibration_vals >= threshold)[0]
        if len(limit_indices) > 0:
            first_fail_idx = limit_indices[0]
            # Bu noktadan sonra ömür 'fiziksel' olarak biter
            y_pred[first_fail_idx:] = 0
    return y_pred


def smooth_predictions(series):
    """ Çift Katmanlı Stabilizasyon: Median + EWMA """
    s = pd.Series(series)
    no_spikes = s.rolling(window=15, min_periods=1, center=True).median()
    smoothed = no_spikes.ewm(alpha=0.05, adjust=False).mean()
    return smoothed.values


def main():
    print("=" * 60)
    print(">>> FİZİKSEL LIMIT KORUMALI ANALİZ VE VİTRİN HAZIRLANIYOR... ⭐")
    print(f">>> ÜS DİZİNİ: {BASE_DIR}")
    print("=" * 60)

    current_results_dir = DEFAULT_RESULTS_DIR
    pred_file = os.path.join(current_results_dir, "test_predictions.npz")

    if not os.path.exists(pred_file):
        print(f"❌ HATA: Tahmin dosyası bulunamadı! Yol: {pred_file}")
        return

    preds = np.load(pred_file)
    y_true_all, y_pred_all = preds['y_true'], preds['y_pred']

    if not os.path.exists(TEST_CSV_PATH):
        print(f"❌ HATA: Test CSV bulunamadı! Yol: {TEST_CSV_PATH}")
        return

    df_test = pd.read_csv(TEST_CSV_PATH)

    current_idx = 0
    plots_data = []
    summary_metrics = []

    for b in df_test['bearing'].unique():
        b_data = df_test[df_test['bearing'] == b]
        expected_windows = len(b_data) - SEQUENCE_LENGTH + 1
        if expected_windows <= 0: continue
        end_idx = current_idx + expected_windows

        if b in SHOWCASE_BEARINGS:
            y_true_b = y_true_all[current_idx: end_idx]
            raw_pred_b = y_pred_all[current_idx: end_idx]

            # 1. Adım: Fiziksel Sınır Kontrolü (20g)
            y_pred_limited = apply_physical_limit(raw_pred_b, b_data)

            # 2. Adım: Ütüleme (Smoothing)
            y_pred_b = smooth_predictions(y_pred_limited)

            # Analitik Hesaplamalar
            mae = mean_absolute_error(y_true_b, y_pred_b)
            rmse = np.sqrt(mean_squared_error(y_true_b, y_pred_b))
            correlation = np.corrcoef(y_true_b, y_pred_b)[0, 1]

            plots_data.append({'name': b, 'y_true': y_true_b, 'y_pred': y_pred_b, 'mae': mae})
            summary_metrics.append([b, f"{mae:.2f} s", f"{rmse:.2f} s", f"%{correlation * 100:.1f}"])

        current_idx = end_idx

    #  ANALİTİK KANIT TABLOSU
    print("\n" + " " * 15 + "📊 MODEL PERFORMANS ANALİZİ (PORTABLE)")
    print("-" * 75)
    print(f"{'Rulman':<15} | {'MAE (Hata)':<15} | {'RMSE':<15} | {'Trend Uyumu (Corr)':<15}")
    print("-" * 75)
    for row in summary_metrics:
        print(f"{row[0]:<15} | {row[1]:<15} | {row[2]:<15} | {row[3]:<15}")
    print("-" * 75)

    # --- POSTER ÇİZİMİ ---
    num_plots = len(plots_data)
    rows = (num_plots + 2) // 3
    plt.figure(figsize=(18, 5 * rows))

    for i, data in enumerate(plots_data):
        ax = plt.subplot(rows, 3, i + 1)
        ax.plot(data['y_true'], label='Gerçek Ömür', color='#1f77b4', linewidth=3, alpha=0.4)
        ax.plot(data['y_pred'], label='AI Tahmini (Stabilize)', color='#2ca02c', linewidth=2.5)
        ax.set_title(f"{data['name']}\nMAE: {data['mae']:.0f} sn", fontsize=13, fontweight='bold')
        ax.set_xlabel("Zaman (Örnek)")
        ax.set_ylabel("RUL (Saniye)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(current_results_dir, "FINAL_PORTABLE_POSTER.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ ANALİTİK GRAFİK HAZIR! Kaydedildi: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()