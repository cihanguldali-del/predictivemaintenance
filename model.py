import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from sklearn.model_selection import train_test_split
import numpy as np
import os

# ================= REPAIRED =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Flaş düzen
DATA_DIR = os.path.join(BASE_DIR, "data", "model_ready_prod_WITH_TEMP")
RESULTS_DIR = os.path.join(DATA_DIR, "training_results_classic")
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "best_hybrid_model_classic.keras")

os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001


# --- Metrik ---
def rul_score(y_true, y_pred):
    error = y_pred - y_true
    # Geç tahmini ağır cezalandıran asimetrik metrik
    score = np.where(error < 0, np.exp(-error / 13) - 1, np.exp(error / 10) - 1)
    return np.sum(score)


def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    score = rul_score(y_true, y_pred)
    return {'MAE': mae, 'RUL_Score': score}


class RULMetricsLogger(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.X_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            y_pred = self.model.predict(self.X_val, verbose=0).flatten()
            m = calculate_metrics(self.y_val, y_pred)
            print(f"   [Epoch {epoch + 1}] Val MAE: {m['MAE']:.2f} | Score: {m['RUL_Score']:.2f}")


def build_hybrid_model(seq_shape, static_shape):
    from tensorflow.keras.regularizers import l2

    # DİNAMİK
    input_seq = Input(shape=seq_shape)
    x = LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001))(input_seq)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = LSTM(64, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # STATİK KOL
    input_static = Input(shape=static_shape)
    y = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(input_static)
    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    # HİBRİT FÜZYON
    combined = Concatenate()([x, y])
    z = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(combined)
    z = Dropout(0.4)(z)
    z = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(z)
    z = Dropout(0.3)(z)
    output = Dense(1, activation='linear')(z)

    model = Model(inputs=[input_seq, input_static], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='huber', metrics=['mae'])
    return model


def main():
    print(f">>> ÜS DİZİNİ: {BASE_DIR}")
    print(">>> MODEL EĞİTİMİ VE OPTİMİZASYONU BAŞLADI... 🚀")

    train_path = os.path.join(DATA_DIR, "prod_train_data.npz")
    test_path = os.path.join(DATA_DIR, "prod_test_data.npz")

    if not os.path.exists(train_path):
        print(f"❌ HATA: Veri bulunamadı! Yol: {train_path}")
        return

    train_data = np.load(train_path)
    test_data = np.load(test_path)

    X_seq_tr, X_seq_val, X_static_tr, X_static_val, y_tr, y_val = train_test_split(
        train_data['X_seq'], train_data['X_static'], train_data['y'], test_size=0.2, random_state=42
    )

    model = build_hybrid_model(X_seq_tr.shape[1:], X_static_tr.shape[1:])

    # --- KRİTİK DÜZELTME: CALLBACK SENKRONİZASYONU ---
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,  # Küçük zıplamaları 'iyileşme' sayma
            patience=10,  # 10 epoch boyunca gerçek düşüş olmazsa kes
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,  # 5 epoch sonunda LR düşür, durmadan önce 1 şans daha ver
            verbose=1
        ),
        RULMetricsLogger(([X_seq_val, X_static_val], y_val))
    ]

    model.fit([X_seq_tr, X_static_tr], y_tr,
              validation_data=([X_seq_val, X_static_val], y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)

    # Test ve Kayıt
    y_pred = model.predict([test_data['X_seq'], test_data['X_static']], verbose=0).flatten()
    np.savez(os.path.join(RESULTS_DIR, 'test_predictions.npz'), y_true=test_data['y'], y_pred=y_pred)
    print(f"✅ EĞİTİM TAMAMLANDI. Model: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()