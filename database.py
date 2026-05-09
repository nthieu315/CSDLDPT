import sqlite3
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import glob
import json
from feature_extractor import extract_features

DB_NAME = "engine_sounds.db"
DATASET_DIR = "dataset_normalized"

def init_db():
    """Khởi tạo CSDL và tạo bảng engine_sounds mới."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Tạo bảng mới, chỉ dùng 1 cột features dạng TEXT (lưu chuỗi JSON)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS engine_sounds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT UNIQUE,
        label TEXT,
        features TEXT
    )
    ''')
    conn.commit()
    return conn

def populate_db(conn):
    """Quét thư mục dataset, trích xuất đặc trưng và lưu vào DB."""
    cursor = conn.cursor()
    
    files = glob.glob(os.path.join(DATASET_DIR, "*.wav"))
    
    if not files:
        print(f"Không tìm thấy file .wav nào trong thư mục '{DATASET_DIR}'. Vui lòng chạy dataset_builder.py trước.")
        return
        
    print(f"Bắt đầu trích xuất đặc trưng (47 chiều) và lưu vào DB cho {len(files)} files...")
    count = 0
    
    for file_path in files:
        # Kiểm tra xem file đã có trong CSDL chưa
        cursor.execute("SELECT id FROM engine_sounds WHERE file_path = ?", (file_path,))
        if cursor.fetchone():
            continue
            
        print(f"Đang xử lý: {os.path.basename(file_path)}")
        features = extract_features(file_path)
        
        if features:
            base_name = os.path.basename(file_path).replace(".wav", "")
            label = "_".join(base_name.split("_")[:-1])
            if not label:
                label = "engine"
            
            # Chuyển đổi list 44 con số thành chuỗi JSON
            features_json = json.dumps(features)
            
            try:
                cursor.execute('''
                INSERT INTO engine_sounds (file_path, label, features)
                VALUES (?, ?, ?)
                ''', (file_path, label, features_json))
                count += 1
            except sqlite3.Error as e:
                print(f"Lỗi DB khi insert {file_path}: {e}")
                
    conn.commit()
    print(f"Hoàn thành! Đã thêm mới {count} records vào CSDL.")

if __name__ == "__main__":
    connection = init_db()
    populate_db(connection)
    connection.close()
