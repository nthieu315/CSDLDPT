import sqlite3
import numpy as np
import json
from feature_extractor import extract_features, N_MFCC
from scipy.spatial.distance import cosine
import sys
import os
from sklearn.preprocessing import StandardScaler
sys.stdout.reconfigure(encoding='utf-8')

DB_NAME = "engine_sounds.db"
OUTPUT_DIR = "ket_qua_trung_gian"


def get_all_features_from_db():
    """Lấy toàn bộ vector đặc trưng từ CSDL."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='engine_sounds'")
    if not cursor.fetchone():
        print("Lỗi: Bảng CSDL chưa được tạo. Hãy chạy database.py trước.")
        conn.close()
        return []
        
    cursor.execute('SELECT id, file_path, label, features FROM engine_sounds')
    rows = cursor.fetchall()
    conn.close()
    
    dataset = []
    for row in rows:
        db_id, file_path, label, features_json = row
        features = np.array(json.loads(features_json), dtype=float)
        dataset.append({
            "id": db_id,
            "file_path": file_path,
            "label": label,
            "features": features
        })
    return dataset


def search_similar_sounds(input_audio_path, top_k=5):
    if not os.path.exists(input_audio_path):
        print(f"Lỗi: Không tìm thấy file đầu vào '{input_audio_path}'")
        return

    print("=" * 60)
    print("HỆ THỐNG TÌM KIẾM TIẾNG ĐỘNG CƠ TƯƠNG ĐỒNG (CBAR)")
    print("=" * 60)
    
    # Bước 1: Trích xuất đặc trưng
    print("\n[Bước 1] Trích xuất đặc trưng (88 chiều) file đầu vào...")
    input_features = extract_features(input_audio_path)
    if input_features is None:
        return
        
    input_vector = np.array(input_features, dtype=float)
    
    # In ra vector đặc trưng (kết quả trung gian cho báo cáo)
    n = N_MFCC  # = 20
    print(f"  -> Vector đặc trưng (88 chiều):")
    print(f"  --- Nhóm 1 | Tần số      [0:3]   ---")
    print(f"     ZCR Mean/Std/Autocorr : {[round(x,6) for x in input_features[0:3]]}")
    print(f"  --- Nhóm 2 | Biên độ     [3:6]   ---")
    print(f"     RMS Mean/Std/Autocorr : {[round(x,6) for x in input_features[3:6]]}")
    print(f"  --- Nhóm 3 | Thời gian   [6:8]   ---")
    print(f"     Tempo                 : {input_features[6]:.2f} BPM")
    print(f"     Silence Ratio         : {input_features[7]:.6f}")
    print(f"  --- Nhóm 4 | Phổ âm      [8:75]  ---")
    print(f"     MFCC Mean    ({n}): {[round(x,4) for x in input_features[8:8+n]]}")
    print(f"     MFCC Std     ({n}): {[round(x,4) for x in input_features[8+n:8+2*n]]}")
    print(f"     MFCC Autocorr({n}): {[round(x,4) for x in input_features[8+2*n:68]]}")
    print(f"     Bandwidth M/S/A       : {[round(x,4) for x in input_features[68:71]]}")
    print(f"     Rolloff   M/S/A       : {[round(x,4) for x in input_features[71:74]]}")
    print(f"     Spectral Contrast     : {input_features[74]:.4f}")
    print(f"  --- Nhóm 5 | Hình dạng  [75:77] ---")
    print(f"     Envelope Std          : {input_features[75]:.6f}")
    print(f"     Pct Above 0.5         : {input_features[76]:.4f}")
    print(f"  --- Nhóm 6 | Độ phức tạp [77:79] ---")
    print(f"     Num Peaks (norm)      : {input_features[77]:.6f}")
    print(f"     Num Valleys (norm)    : {input_features[78]:.6f}")
    print(f"  --- Nhóm 7 | Biên độ màu [79:82] ---")
    print(f"     Low (<2kHz)           : {input_features[79]:.4f}")
    print(f"     Mid (2-4kHz)          : {input_features[80]:.4f}")
    print(f"     High (>4kHz)          : {input_features[81]:.4f}")
    print(f"  --- Nhóm 8 | Độ chói     [82:86] ---")
    print(f"     Centroid M/S/A        : {[round(x,4) for x in input_features[82:85]]}")
    print(f"     Spectral Flatness     : {input_features[85]:.6f}")
    print(f"  --- Nhóm 9 | Độ chạy     [86]    ---")
    print(f"     Attack Time (s)       : {input_features[86]:.4f}")
    print(f"  --- Nhóm 10| Độ suy giảm [87]    ---")
    print(f"     Decay Time (s)        : {input_features[87]:.4f}")
    
    # Bước 2: Đọc CSDL
    print(f"\n[Bước 2] Đọc dữ liệu từ CSDL ({DB_NAME})...")
    db_data = get_all_features_from_db()
    if not db_data:
        return
    print(f"  -> Đã nạp {len(db_data)} bản ghi từ CSDL.")
        
    # Bước 3: Chuẩn hóa StandardScaler
    db_vectors = np.array([item["features"] for item in db_data])

    # ================================================================
    # TRỌNG SỐ TỪNG NHÓM — Chỉnh tại đây khi cần tinh chỉnh
    # (Ghi chú: các giá trị này sẽ được chuẩn hóa tự động theo tổng khi tính)
    # ================================================================
    GROUPS = [
        ("Tần số",      slice(0, 3),   0.033),  # Nhóm 1
        ("Biên độ",     slice(3, 6),   0.033),  # Nhóm 2
        ("Thời gian",   slice(6, 8),   0.050),  # Nhóm 3
        ("Phổ âm",      slice(8, 75),  0.400),  # Nhóm 4
        ("Hình dạng",   slice(75, 77), 0.033),  # Nhóm 5
        ("Độ phức tạp", slice(77, 79), 0.050),  # Nhóm 6
        ("Biên độ màu", slice(79, 82), 0.100),  # Nhóm 7
        ("Độ chói",     slice(82, 86), 0.150),  # Nhóm 8
        ("Độ chạy",     slice(86, 87), 0.075),  # Nhóm 9
        ("Độ suy giảm", slice(87, 88), 0.075),  # Nhóm 10
    ]   # Tổng trọng số = 1.0
    # ================================================================

    print(f"\n[Bước 3] Chuẩn hóa StandardScaler (chuẩn hóa từng chiều về cùng tháng đo)...")
    scaler = StandardScaler()
    db_vectors_scaled = scaler.fit_transform(db_vectors)
    input_vector_scaled = scaler.transform([input_vector])[0]

    # Hàm tính cosine similarity trên một nhóm
    def group_cosine(a, b):
        """Cosine similarity giữa 2 vector con, trả về [0,1]."""
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(max(0.0, np.dot(a, b) / (na * nb)))

    # Bước 4: Tính Group-wise Cosine Similarity
    print(f"\n[Bước 4] Tính Cosine Similarity theo từng nhóm rồi tổng hợp có trọng số...")
    print(f"   Công thức: Sim_final = Σ(w_i × Sim_Group_i), tổng w = {sum(g[2] for g in GROUPS):.3f}")
    results = []

    for i, item in enumerate(db_data):
        db_vec = db_vectors_scaled[i]
        sim_final = 0.0
        for _name, slc, w in GROUPS:
            sim_final += w * group_cosine(input_vector_scaled[slc], db_vec[slc])

        results.append({
            "file_path": item["file_path"],
            "label": item["label"],
            "similarity": round(sim_final, 6)
        })
        
    # Bước 5: Xếp hạng và trả kết quả
    results.sort(key=lambda x: x["similarity"], reverse=True)
    top_results = results[:top_k]
    
    print(f"\n[Bước 5] Xếp hạng và trả Top-{top_k} kết quả:")
    print("\n" + "=" * 60)
    print(f"  KẾT QUẢ TÌM KIẾM TOP {top_k} FILE TƯƠNG ĐỒNG NHẤT")
    print("=" * 60)
    for i, res in enumerate(top_results):
        print(f"  Hạng {i+1}:")
        print(f"    File : {res['file_path']}")
        print(f"    Label: {res['label']}")
        print(f"    Cosine Similarity: {res['similarity']:.4f}")
        print("  " + "-" * 40)
    
    
    print("\n" + "=" * 60)
    print("HOÀN TẤT QUÁ TRÌNH TÌM KIẾM.")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Sử dụng: py search_engine.py <đường_dẫn_file_âm_thanh>")
    else:
        input_file = sys.argv[1]
        search_similar_sounds(input_file)
