import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import json

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GRU, Dense, Dropout, GaussianNoise, Input
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ======================================
# 1) LOAD DATA
# ======================================
print("📂 Đang tải dữ liệu...")

X_train_full = np.load("Data/X_train.npy")
X_test        = np.load("Data/X_test.npy")
y_train_full  = np.load("Data/y_train.npy")
y_test        = np.load("Data/y_test.npy")
scaler        = joblib.load("Data/gru_scaler.pkl")

print("✅ Đã load scaler")
print(f"X_train_full : {X_train_full.shape}")
print(f"X_test       : {X_test.shape}")
print(f"y_train_full : {y_train_full.shape}")
print(f"y_test       : {y_test.shape}")


# ======================================
# 2) SPLIT VALIDATION THEO THỜI GIAN
#    (KHÔNG shuffle — time series!)
# ======================================
# Dùng 85% đầu để train, 15% cuối để validate
# Giữ đúng thứ tự chronological
VAL_RATIO = 0.15
split_idx  = int(len(X_train_full) * (1 - VAL_RATIO))

X_train = X_train_full[:split_idx]
X_val   = X_train_full[split_idx:]
y_train = y_train_full[:split_idx]
y_val   = y_train_full[split_idx:]

print(f"\n📊 Chronological split:")
print(f"  Train : {X_train.shape[0]:,} mẫu")
print(f"  Val   : {X_val.shape[0]:,} mẫu  ({VAL_RATIO*100:.0f}% cuối)")
print(f"  Test  : {X_test.shape[0]:,} mẫu")


# ======================================
# 3) BUILD MODEL (đã giảm capacity + L2)
# ======================================
print("\n🧠 Đang build mô hình GRU...")

REG = l2(1e-4)  # L2 regularization — kiểm soát trọng số phình to

model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),

    # Gaussian noise để augment — giúp model robust với nhiễu đo lường
    GaussianNoise(0.01),

    # Layer 1: giảm từ 64 → 32 units để tránh học vẹt chi tiết nhiễu
    GRU(
        32,
        return_sequences=True,
        kernel_regularizer=REG,
        recurrent_regularizer=REG
    ),
    Dropout(0.4),

    # Layer 2: giảm từ 32 → 16 units
    GRU(
        16,
        kernel_regularizer=REG,
        recurrent_regularizer=REG
    ),
    Dropout(0.4),

    # Bỏ Dense(16) trung gian — không cần thiết cho task đơn giản này
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=tf.keras.losses.Huber(),
    metrics=["mae"]
)

model.summary()


# ======================================
# 4) CALLBACKS
# ======================================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,           # Tăng từ 8 → 15 để không dừng quá sớm
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "Data/best_tidal_gru.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=0
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,            # Tăng từ 3 → 5
    min_lr=1e-6,
    verbose=1
)


# ======================================
# 5) TRAIN
#    shuffle=False — BẮT BUỘC với time series
#    validation_data thay vì validation_split
# ======================================
BATCH_SIZE = 1024

print("\n🚀 Bắt đầu training...")
print("⚠️  shuffle=False — giữ thứ tự thời gian cho time series\n")

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),   # Split thủ công theo thời gian
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1,
    shuffle=False                      # KHÔNG shuffle time series!
)


# ======================================
# 6) VẼ METRIC THEO EPOCH
# ======================================
print("\n📈 Đang vẽ metric theo epoch...")

epochs_ran = range(1, len(history.history["loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(epochs_ran, history.history["loss"],     label="Train Loss", color="steelblue")
axes[0].plot(epochs_ran, history.history["val_loss"], label="Val Loss",   color="tomato")
axes[0].set_title("Huber Loss theo Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Huber Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAE
axes[1].plot(epochs_ran, history.history["mae"],     label="Train MAE", color="steelblue")
axes[1].plot(epochs_ran, history.history["val_mae"], label="Val MAE",   color="tomato")
axes[1].set_title("MAE theo Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MAE")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Data/training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# Chẩn đoán overfitting
best_epoch    = np.argmin(history.history["val_loss"]) + 1
best_val_loss = np.min(history.history["val_loss"])
best_trn_loss = history.history["loss"][best_epoch - 1]
gap           = best_val_loss - best_trn_loss

print(f"\n🏆 Best Epoch    : {best_epoch}")
print(f"📉 Train Loss    : {best_trn_loss:.6f}")
print(f"📉 Val Loss      : {best_val_loss:.6f}")
print(f"📊 Gap (Val-Trn) : {gap:.6f}  ", end="")
if gap < 0.001:
    print("✅ Tốt — không overfitting đáng kể")
elif gap < 0.005:
    print("⚠️  Nhẹ — có thể chấp nhận được")
else:
    print("❌ Overfitting vẫn còn — thử giảm units thêm hoặc tăng L2")


# ======================================
# 7) TEST EVALUATION (scaled values)
# ======================================
print("\n🧪 Đánh giá tập test (scaled)...")
y_pred = model.predict(X_test, batch_size=BATCH_SIZE)

rmse_scaled = np.sqrt(mean_squared_error(y_test, y_pred))
mae_scaled  = mean_absolute_error(y_test, y_pred)

print(f"✅ Test RMSE (scaled): {rmse_scaled:.6f}")
print(f"✅ Test MAE  (scaled): {mae_scaled:.6f}")


# ======================================
# 8) INVERSE TRANSFORM → ĐƠN VỊ MÉT
# ======================================
print("\n📏 Chuyển đổi về đơn vị Mét...")

def inverse_first_col(y_data, scaler, n_features):
    """Inverse transform cột đầu tiên (mực nước) từ scaled → mét."""
    dummy = np.zeros((len(y_data), n_features))
    dummy[:, 0] = y_data.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

n_feat     = X_train.shape[2]
y_test_m   = inverse_first_col(y_test, scaler, n_feat)
y_pred_m   = inverse_first_col(y_pred, scaler, n_feat)

final_rmse = np.sqrt(mean_squared_error(y_test_m, y_pred_m))
final_mae  = mean_absolute_error(y_test_m, y_pred_m)
final_r2   = r2_score(y_test_m, y_pred_m)

print(f"\n📊 --- KẾT QUẢ THỰC TẾ (Đơn vị: Mét) ---")
print(f"✅ RMSE : {final_rmse:.4f} m")
print(f"✅ MAE  : {final_mae:.4f} m")
print(f"✅ R²   : {final_r2:.4f}  ({final_r2*100:.2f}%)")


# ======================================
# 9) LƯU PREDICTION CSV
# ======================================
pred_df = pd.DataFrame({
    "actual":    y_test_m,
    "predicted": y_pred_m,
    "residual":  y_test_m - y_pred_m
})
pred_df.to_csv("Data/gru_predictions.csv", index=False)
print("\n💾 Đã lưu: Data/gru_predictions.csv")


# ======================================
# 10) BIỂU ĐỒ SO SÁNH (ĐƠN VỊ MÉT)
# ======================================
N_PLOT = 500

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Actual vs Predicted
axes[0].plot(y_test_m[:N_PLOT],  label="Thực tế (Vũng Tàu)", color="steelblue", linewidth=1.5)
axes[0].plot(y_pred_m[:N_PLOT],  label="Dự báo (GRU)",        color="tomato",    linewidth=1.5, linestyle="--")
axes[0].set_title(f"So sánh Mực nước thực tế và Dự báo ({N_PLOT} điểm đầu)")
axes[0].set_xlabel("Mốc thời gian (Phút)")
axes[0].set_ylabel("Mực nước (m)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Residual
residual = y_test_m[:N_PLOT] - y_pred_m[:N_PLOT]
axes[1].plot(residual, color="purple", linewidth=0.8, alpha=0.7)
axes[1].axhline(0, color="black", linewidth=1)
axes[1].axhline( final_rmse, color="tomato",    linewidth=1, linestyle="--", label=f"+RMSE ({final_rmse:.3f}m)")
axes[1].axhline(-final_rmse, color="steelblue", linewidth=1, linestyle="--", label=f"-RMSE ({final_rmse:.3f}m)")
axes[1].set_title("Residual Error (Actual − Predicted)")
axes[1].set_xlabel("Mốc thời gian")
axes[1].set_ylabel("Sai số (m)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Error distribution
errors = y_test_m - y_pred_m
axes[2].hist(errors, bins=60, color="skyblue", edgecolor="black", alpha=0.7)
axes[2].axvline(0,             color="red",   linewidth=2, linestyle="--", label="Zero")
axes[2].axvline(errors.mean(), color="green", linewidth=1.5, linestyle="-", label=f"Mean={errors.mean():.4f}m")
axes[2].set_title("Phân phối sai số (m)")
axes[2].set_xlabel("Sai số (m)")
axes[2].set_ylabel("Số lượng mẫu")
axes[2].legend()
axes[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("Data/evaluation_plots.png", dpi=150, bbox_inches="tight")
plt.show()


# ======================================
# 11) LƯU HISTORY
# ======================================
np.save("Data/train_loss.npy", history.history["loss"])
np.save("Data/val_loss.npy",   history.history["val_loss"])
np.save("Data/train_mae.npy",  history.history["mae"])
np.save("Data/val_mae.npy",    history.history["val_mae"])
print("💾 Đã lưu training history")


# ======================================
# 12) LƯU CONFIG CHO WEB DEPLOY
# ======================================
deploy_config = {
    "model_version": "2.0-fixed",
    "seq_length":    int(X_train.shape[1]),
    "num_features":  int(X_train.shape[2]),
    "feature_cols":  ["ra2(m)", "sin_day", "cos_day", "lag_120"],
    "val_split":     "chronological_last_15pct",
    "best_epoch":    int(best_epoch),
    "rmse_scaled":   float(rmse_scaled),
    "mae_scaled":    float(mae_scaled),
    "rmse_met":      float(final_rmse),
    "mae_met":       float(final_mae),
    "r2_score":      float(final_r2),
    "train_samples": int(X_train.shape[0]),
    "val_samples":   int(X_val.shape[0]),
    "test_samples":  int(X_test.shape[0]),
}

with open("Data/gru_config.json", "w") as f:
    json.dump(deploy_config, f, indent=4, ensure_ascii=False)

print("🌐 Đã lưu config deploy: Data/gru_config.json")
print("\n🏁 HOÀN TẤT! Các thay đổi chính so với phiên bản cũ:")
print("   ✅ shuffle=False — giữ thứ tự thời gian")
print("   ✅ Validation split theo chronological order")
print("   ✅ L2 regularization trên cả 2 GRU layers")
print("   ✅ GaussianNoise(0.01) để augment")
print("   ✅ Giảm capacity: GRU(32→16), bỏ Dense(16)")
print("   ✅ Early stopping patience tăng lên 15")
print("   ✅ Tự động chẩn đoán train/val gap")