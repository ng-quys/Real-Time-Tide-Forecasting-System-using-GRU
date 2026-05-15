import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib  # <--- Thêm thư viện này để lưu Scaler

# Cấu hình
TIDAL_CYCLE_MINUTES = 745
SEQ_LENGTH = 120
TRAIN_RATIO = 0.8
GAP_LINEAR_LIMIT = 15  
GAP_SEASONAL_LIMIT = 120 

# ======================================
# VẼ DỮ LIỆU CHUYÊN NGHIỆP
# ======================================
def plot_data(df_or_segments, title, is_segments=False, sample_points=5000):
    plt.figure(figsize=(15, 5))
    if not is_segments:
        plt.plot(df_or_segments['Time (UTC)'][:sample_points], df_or_segments['ra2(m)'][:sample_points])
    else:
        count = 0
        for seg in df_or_segments:
            if count < sample_points:
                plt.plot(seg['Time (UTC)'], seg['ra2(m)'], color='tab:blue')
                count += len(seg)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Sea Level (m)")
    plt.grid(True, alpha=0.3)
    plt.show()

# ======================================
# TIỀN XỬ LÝ GAP-AWARE
# ======================================
def preprocess_tidal_data(file_path: str):
    print("🚀 Đang xử lý dữ liệu...")
    df = pd.read_csv(file_path)
    df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'])
    df['ra2(m)'] = pd.to_numeric(df['ra2(m)'], errors='coerce')
    df = df.sort_values('Time (UTC)').drop_duplicates('Time (UTC)')
    
    df = df.set_index('Time (UTC)').resample('1min').mean()
    
    missing_mask = df['ra2(m)'].isna()
    groups = (missing_mask != missing_mask.shift()).cumsum()
    
    segments = []
    current_segment_blocks = []

    for _, block in df.groupby(groups):
        is_missing = block['ra2(m)'].isna().all()
        
        if not is_missing:
            current_segment_blocks.append(block)
            continue
        
        gap_size = len(block)
        
        if gap_size <= GAP_LINEAR_LIMIT:
            filled = block.copy()
            filled['ra2(m)'] = df['ra2(m)'].interpolate(method='linear').loc[block.index]
            current_segment_blocks.append(filled)
            
        elif gap_size <= GAP_SEASONAL_LIMIT:
            filled = block.copy()
            seasonal_vals = []
            for t in block.index:
                prev = t - pd.Timedelta(minutes=TIDAL_CYCLE_MINUTES)
                nxt = t + pd.Timedelta(minutes=TIDAL_CYCLE_MINUTES)
                v1 = df.loc[prev, 'ra2(m)'] if prev in df.index else np.nan
                v2 = df.loc[nxt, 'ra2(m)'] if nxt in df.index else np.nan
                val = np.nanmean([v1, v2]) if not np.isnan([v1, v2]).all() else np.nan
                seasonal_vals.append(val)
            
            filled['ra2(m)'] = seasonal_vals
            filled['ra2(m)'] = filled['ra2(m)'].interpolate().ffill().bfill()
            current_segment_blocks.append(filled)
            
        else:
            if current_segment_blocks:
                segments.append(pd.concat(current_segment_blocks).reset_index())
                current_segment_blocks = []
                
    if current_segment_blocks:
        segments.append(pd.concat(current_segment_blocks).reset_index())
        
    return segments

# ======================================
# TẠO SEQUENCE & LƯU SCALER
# ======================================
def prepare_training_data(segments):
    print("🧠 Đang tạo dữ liệu huấn luyện...")
    
    processed_segments = []
    for seg in segments:
        s = seg.copy()
        s['hour_sin'] = np.sin(2 * np.pi * s['Time (UTC)'].dt.hour / 24)
        s['hour_cos'] = np.cos(2 * np.pi * s['Time (UTC)'].dt.hour / 24)
        processed_segments.append(s)

    total_len = sum(len(s) for s in processed_segments)
    train_size = int(total_len * TRAIN_RATIO)
    
    train_segments = []
    test_segments = []
    current_accumulated = 0
    
    for s in processed_segments:
        if current_accumulated < train_size:
            train_segments.append(s)
        else:
            test_segments.append(s)
        current_accumulated += len(s)

    # 3. Fit Scaler & LƯU LẠI
    scaler = MinMaxScaler()
    features = ['ra2(m)', 'hour_sin', 'hour_cos']
    full_train_df = pd.concat(train_segments)
    scaler.fit(full_train_df[features])
    
    # LƯU SCALER TẠI ĐÂY
    joblib.dump(scaler, "Data/gru_scaler.pkl")
    print("💾 Đã lưu Scaler vào Data/gru_scaler.pkl")

    def create_windows(segment_list):
        X, y = [], []
        for s in segment_list:
            if len(s) <= SEQ_LENGTH: continue
            scaled_data = scaler.transform(s[features])
            for i in range(len(scaled_data) - SEQ_LENGTH):
                X.append(scaled_data[i : i+SEQ_LENGTH])
                y.append(scaled_data[i+SEQ_LENGTH, 0])
        return np.array(X, dtype='float32'), np.array(y, dtype='float32')

    X_train, y_train = create_windows(train_segments)
    X_test, y_test = create_windows(test_segments)
    
    return X_train, X_test, y_train, y_test, scaler

# ======================================
# CHẠY CHƯƠNG TRÌNH
# ======================================
if __name__ == '__main__':
    data_path = 'Data/vungtau_full.csv'
    if not os.path.exists('Data'): os.makedirs('Data')

    segments = preprocess_tidal_data(data_path)
    
    # Vẽ kiểm tra
    plot_data(segments, "Processed Sea Level (Clean Segments)", is_segments=True)

    # Chuẩn bị dữ liệu và lưu scaler
    X_train, X_test, y_train, y_test, scaler = prepare_training_data(segments)

    # Lưu dữ liệu numpy
    np.save('Data/X_train.npy', X_train)
    np.save('Data/X_test.npy', X_test)
    np.save('Data/y_train.npy', y_train)
    np.save('Data/y_test.npy', y_test)
    
    print(f"📊 Dataset Ready: Train={X_train.shape}, Test={X_test.shape}")