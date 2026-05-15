import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# ======================================
# 1. LOAD DỮ LIỆU VÀ MODEL
# ======================================
print("📂 Đang tải dữ liệu và mô hình tốt nhất...")
X_test = np.load("Data/X_test.npy")
y_test = np.load("Data/y_test.npy")
scaler = joblib.load("Data/gru_scaler.pkl")

# Load model với định dạng mới của Keras
model = load_model("Data/best_tidal_gru.keras")

# ======================================
# 2. DỰ BÁO (KHẮC PHỤC LỖI BATCH_OUTPUTS)
# ======================================
print("🔮 Đang thực hiện dự báo trên tập Test...")
try:
    # Chỉ định batch_size rõ ràng để tránh lỗi iterator trên Keras 3
    y_pred = model.predict(X_test, batch_size=512, verbose=1)
except Exception as e:
    print(f"⚠️ Phát hiện lỗi GPU: {e}. Đang chuyển sang CPU để dự báo...")
    with tf.device('/CPU:0'):
        y_pred = model.predict(X_test, batch_size=512, verbose=1)

# ======================================
# 3. CHUYỂN ĐỔI VỀ ĐƠN VỊ MÉT (INVERSE TRANSFORM)
# ======================================
def get_inverse_met(y_data, scaler):
    # Lấy số lượng feature mà scaler đã fit (thường là 3: ra2, hour_sin, hour_cos)
    num_features = scaler.n_features_in_
    dummy = np.zeros((len(y_data), num_features))
    dummy[:, 0] = y_data.flatten()
    # Chỉ lấy cột đầu tiên sau khi inverse (cột ra2 - mực nước)
    return scaler.inverse_transform(dummy)[:, 0]

y_test_met = get_inverse_met(y_test, scaler)
y_pred_met = get_inverse_met(y_pred, scaler)

# ======================================
# 4. TÍNH TOÁN CÁC CHỈ SỐ THỐNG KÊ (KPI)
# ======================================
rmse = np.sqrt(mean_squared_error(y_test_met, y_pred_met))
mae = mean_absolute_error(y_test_met, y_pred_met)
r2 = r2_score(y_test_met, y_pred_met)

print("\n" + "="*40)
print("🏆 KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH GRU (VŨNG TÀU)")
print(f"📍 RMSE (Sai số căn lề): {rmse:.4f} m")
print(f"📍 MAE (Sai số tuyệt đối): {mae:.4f} m")
print(f"📍 R² Score (Độ tương quan): {r2:.4f} ({r2*100:.2f}%)")
print("="*40)

# ======================================
# 5. TRỰC QUAN HÓA KẾT QUẢ
# ======================================

# Biểu đồ 1: So sánh Thực tế vs Dự báo (2 ngày đầu - 2880 phút)
plt.figure(figsize=(15, 6))
plt.plot(y_test_met[:1440], label="Thực tế", color='#1f77b4', linewidth=2, alpha=0.8)
plt.plot(y_pred_met[:1440], label="Dự báo GRU", color='#d62728', linestyle='--', linewidth=2)
plt.title("So sánh Mực nước Thực tế vs Dự báo (24 Giờ đầu tập Test)", fontsize=14)
plt.xlabel("Thời gian (Phút)", fontsize=12)
plt.ylabel("Mực nước (m)", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("Data/plot_comparison.png")
plt.show()

# Biểu đồ 2: Phân tích sai số dư (Residual Analysis)
residuals = y_test_met - y_pred_met
plt.figure(figsize=(15, 5))
plt.scatter(range(len(residuals[:1000])), residuals[:1000], color='purple', alpha=0.4, s=8)
plt.axhline(0, color='black', linestyle='-', linewidth=1)
plt.title("Phân tích sai số dư (Residuals - 1000 mẫu đầu)", fontsize=14)
plt.xlabel("Mẫu dữ liệu", fontsize=12)
plt.ylabel("Sai số (m)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Data/plot_residuals.png")
plt.show()

# Biểu đồ 3: Histogram phân phối sai số
plt.figure(figsize=(10, 6))
plt.hist(errors := residuals, bins=60, color='seagreen', alpha=0.7, edgecolor='white')
plt.axvline(0, color='red', linestyle='dashed', linewidth=2, label='Dự báo khớp')
plt.title("Phân phối sai số thực tế (m)", fontsize=14)
plt.xlabel("Sai lệch (m)", fontsize=12)
plt.ylabel("Tần suất", fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("Data/plot_error_dist.png")
plt.show()

# ======================================
# 6. LƯU KẾT QUẢ CHI TIẾT
# ======================================
output_df = pd.DataFrame({
    'Actual_m': y_test_met,
    'Predicted_m': y_pred_met,
    'Error_m': residuals
})
output_df.to_csv("Data/final_results_evaluation.csv", index=False)

print("\n💾 Đã lưu 3 biểu đồ và file CSV kết quả vào thư mục 'Data/'.")
print("🏁 Quy trình đánh giá hoàn tất!")