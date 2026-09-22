import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

# =================GÖRECELİ YOLLAR =================
# Kodun çalıştığı klasörü otomatik bul
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Verileri 'data' klasöründen al, sonuçları 'model_ready_prod' klasörüne yaz
DATA_DIR = os.path.join(BASE_DIR, "data", "model_ready")

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "model_ready_prod_WITH_TEMP")

# Klasör yoksa oluştur
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQUENCE_LENGTH = 30
CONDITION_PARAMS = {1: (1800, 4000), 2: (1650, 4200), 3: (1500, 5000)}


def get_numeric_features(df, exclude_cols):
    candidates = [c for c in df.columns if c not in exclude_cols]
    try:
        df[candidates] = df[candidates].apply(pd.to_numeric, errors='raise')
    except:
        pass
    return candidates


def create_dual_inputs_prod(df, scaler_static=None, enforced_cols=None, seq_len=30, mode='train'):
    loads, speeds = [], []
    for idx, row in df.iterrows():
        try:
            s, l = CONDITION_PARAMS[int(row.get('condition'))]
            loads.append(l);
            speeds.append(s)
        except:
            continue
    static_matrix = np.column_stack((speeds, loads))

    if scaler_static is None:
        scaler_static = MinMaxScaler()
        static_matrix = scaler_static.fit_transform(static_matrix)
    else:
        static_matrix = scaler_static.transform(static_matrix)

    dynamic_matrix = df[enforced_cols].values
    bearings = df['bearing'].unique()
    y = [] if mode == 'train' else None
    X_seq, X_static = [], []

    for b in bearings:
        indices = np.where(df['bearing'].values == b)[0]
        b_dynamic = dynamic_matrix[indices]
        b_static = static_matrix[indices]
        if len(b_dynamic) < seq_len: continue

        for i in range(len(b_dynamic) - seq_len + 1):
            X_seq.append(b_dynamic[i: i + seq_len])
            X_static.append(b_static[i + seq_len - 1])
            if mode == 'train':
                y.append(df['rul_seconds'].values[indices[i + seq_len - 1]])

    return np.array(X_seq), np.array(X_static), (np.array(y) if mode == 'train' else None), scaler_static


def main():
    print(f">>> ÇALIŞMA DİZİNİ: {BASE_DIR}")
    print(">>> ADIM 2: ESKİ 3D PAKETLEME (SICAKLIK DAHİL)...")

    train_path = os.path.join(DATA_DIR, "ELITE_learning_dataset_final.csv")
    test_path = os.path.join(DATA_DIR, "ELITE_test_dataset_final.csv")

    if not os.path.exists(train_path):
        print(f"❌ HATA: Veri bulunamadı! Lütfen {DATA_DIR} klasörünü kontrol edin.")
        return

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    exclude = ['condition', 'bearing', 'subset', 'rul_seconds', 'temp_flag']
    TRAIN_FEATS = get_numeric_features(df_train, exclude)
    _ = get_numeric_features(df_test, exclude)

    print(f"   Özellik Sayısı: {len(TRAIN_FEATS)} (Sıcaklık: {'temperature' in TRAIN_FEATS})")

    scaler_dynamic = MinMaxScaler()
    df_train[TRAIN_FEATS] = scaler_dynamic.fit_transform(df_train[TRAIN_FEATS])
    df_test[TRAIN_FEATS] = scaler_dynamic.transform(df_test[TRAIN_FEATS])

    X_seq_train, X_static_train, y_train, scaler_static = create_dual_inputs_prod(
        df_train, scaler_static=None, enforced_cols=TRAIN_FEATS, seq_len=SEQUENCE_LENGTH, mode='train'
    )
    X_seq_test, X_static_test, y_test, _ = create_dual_inputs_prod(
        df_test, scaler_static=scaler_static, enforced_cols=TRAIN_FEATS, seq_len=SEQUENCE_LENGTH, mode='train'
    )

    np.savez(os.path.join(OUTPUT_DIR, "prod_train_data.npz"), X_seq=X_seq_train, X_static=X_static_train, y=y_train)
    np.savez(os.path.join(OUTPUT_DIR, "prod_test_data.npz"), X_seq=X_seq_test, X_static=X_static_test, y=y_test)

    print(f"✅ ADIM 2 TAMAM. Dosyalar şuraya yazıldı: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()