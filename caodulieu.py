import pandas as pd
import requests
import io
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================
# CONFIG
# =========================
STATION_CODE = "vung"
START_DATE = datetime(2024, 6, 1)
END_DATE = datetime.utcnow()
STEP_DAYS = 7
OUTPUT_CSV = "vungtau_full_from_2024_06_01.csv"


def create_session():
    session = requests.Session()

    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504]
    )

    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)

    return session


def fetch_chunk(session, end_time, period_days=30):
    url = (
        f"https://www.ioc-sealevelmonitoring.org/bgraph.php"
        f"?code={STATION_CODE}&output=tab"
        f"&period={period_days}"
        f"&endtime={end_time.strftime('%Y-%m-%d')}"
    )

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = session.get(url, headers=headers, timeout=120)
    response.raise_for_status()

    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0]

    # clean header
    df.columns = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop=True)

    df = df[['Time (UTC)', 'ra2(m)']].copy()
    df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'])
    df['ra2(m)'] = pd.to_numeric(df['ra2(m)'], errors='coerce')

    return df


# def crawl_full_data():
#     session = create_session()
#     all_chunks = []

#     current_end = START_DATE + timedelta(days=STEP_DAYS)

#     while current_end <= END_DATE + timedelta(days=1):
#         print(f"Đang tải block kết thúc tại: {current_end.date()}")

#         try:
#             chunk = fetch_chunk(session, current_end, STEP_DAYS)
#             all_chunks.append(chunk)
#         except Exception as e:
#             print(f"Lỗi block {current_end.date()}: {e}")

#         current_end += timedelta(days=STEP_DAYS)

#     full_df = pd.concat(all_chunks, ignore_index=True)

#     # xóa trùng
#     full_df = full_df.drop_duplicates(subset='Time (UTC)')
#     full_df = full_df.sort_values('Time (UTC)').reset_index(drop=True)

#     return full_df

def crawl_full_data():
    session = create_session()
    all_chunks = []

    current_end = START_DATE + timedelta(days=STEP_DAYS)

    while current_end <= END_DATE + timedelta(days=1):
        print(f"Đang tải block kết thúc tại: {current_end.date()}")

        try:
            chunk = fetch_chunk(session, current_end, STEP_DAYS)
            print(f"✅ Block {current_end.date()} tải xong, {len(chunk)} dòng")
            print(chunk.head(3))  # in 3 dòng đầu tiên để xem dữ liệu
            all_chunks.append(chunk)
        except Exception as e:
            print(f"❌ Lỗi block {current_end.date()}: {e}")

        current_end += timedelta(days=STEP_DAYS)

    full_df = pd.concat(all_chunks, ignore_index=True)

    # xóa trùng
    full_df = full_df.drop_duplicates(subset='Time (UTC)')
    full_df = full_df.sort_values('Time (UTC)').reset_index(drop=True)

    return full_df

def make_continuous(df):
    full_time = pd.date_range(
        start=df['Time (UTC)'].min(),
        end=df['Time (UTC)'].max(),
        freq='1min'
    )

    full_df = pd.DataFrame({'Time (UTC)': full_time})
    df = full_df.merge(df, on='Time (UTC)', how='left')

    return df


def main():
    df = crawl_full_data()

    print("\nĐang tạo timeline liên tục...")
    df = make_continuous(df)

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Đã lưu: {OUTPUT_CSV}")
    print(f"Tổng số dòng: {len(df)}")
    print(f"Số giá trị thiếu: {df['ra2(m)'].isna().sum()}")


if __name__ == "__main__":
    main()