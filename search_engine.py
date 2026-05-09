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

# Kiểm tra xem matplotlib có sẵn không (tùy chọn, dùng để vẽ biểu đồ kết quả trung gian)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

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

def generate_intermediate_results(input_audio_path, input_features, top_results):
    """
    Sinh ra các kết quả trung gian (Yêu cầu 4b):
    - Waveform của file đầu vào
    - Mel-Spectrogram của file đầu vào
    - Biểu đồ vector đặc trưng 47 chiều
    - Biểu đồ Similarity Score của Top-5
    Lưu tất cả vào thư mục ket_qua_trung_gian/
    """
    if not HAS_MATPLOTLIB:
        print("  (Bỏ qua vẽ biểu đồ — cài matplotlib để kích hoạt: pip install matplotlib)")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    y, sr = librosa.load(input_audio_path, sr=None)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Ket qua trung gian - {os.path.basename(input_audio_path)}', fontsize=16, fontweight='bold')
    
    # ========== 1. Waveform ==========
    ax1 = axes[0, 0]
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='#2196F3')
    ax1.set_title('1. Waveform (Dang song thoi gian)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Thoi gian (s)')
    ax1.set_ylabel('Bien do')
    ax1.grid(True, alpha=0.3)
    
    # ========== 2. Mel-Spectrogram ==========
    ax2 = axes[0, 1]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=160, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, hop_length=160, x_axis='time', y_axis='mel', ax=ax2, cmap='magma')
    ax2.set_title('2. Mel-Spectrogram (Pho tan so)', fontsize=12, fontweight='bold')
    fig.colorbar(img, ax=ax2, format='%+2.0f dB')
    
    # ========== 3. Feature Vector 47 chiều ==========
    ax3 = axes[1, 0]
    n_mfcc = N_MFCC
    colors = (
        ['#1976D2'] * n_mfcc +   # MFCC Mean
        ['#7B1FA2'] * n_mfcc +   # MFCC Std
        ['#5E35B1'] * n_mfcc +   # MFCC Autocorr
        ['#E64A19'] * 5  +       # Spec Mean
        ['#FF7043'] * 5  +       # Spec Std
        ['#AB47BC'] * 5  +       # Autocorr
        ['#26A69A'] * 3  +       # Energy Bands
        ['#FFA726'] * 3          # Tempo/Env/Silence
    )
    x_pos = np.arange(len(input_features))
    ax3.bar(x_pos, input_features, color=colors, width=0.8)
    ax3.set_title(f'3. Feature Vector ({3*N_MFCC+21} chieu)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Chi so dac trung')
    ax3.set_ylabel('Gia tri')
    ax3.set_xticks(x_pos[::5])
    ax3.grid(True, alpha=0.3, axis='y')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1976D2', label=f'MFCC Mean ({N_MFCC})'),
        Patch(facecolor='#7B1FA2', label=f'MFCC Std ({N_MFCC})'),
        Patch(facecolor='#5E35B1', label=f'MFCC Autocorr ({N_MFCC})'),
        Patch(facecolor='#E64A19', label='Spec Mean (5)'),
        Patch(facecolor='#FF7043', label='Spec Std (5)'),
        Patch(facecolor='#AB47BC', label='Autocorr (5)'),
        Patch(facecolor='#26A69A', label='Energy Bands (3)'),
        Patch(facecolor='#FFA726', label='Tempo/Env/SR (3)'),
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=7)
    
    # ========== 4. Similarity Scores Top-5 ==========
    ax4 = axes[1, 1]
    labels_top = [os.path.basename(r['file_path']) for r in top_results]
    scores_top = [r['similarity'] for r in top_results]
    bar_colors = ['#4CAF50' if s >= 0.8 else '#FFC107' if s >= 0.5 else '#F44336' for s in scores_top]
    bars = ax4.barh(range(len(labels_top)), scores_top, color=bar_colors)
    ax4.set_yticks(range(len(labels_top)))
    ax4.set_yticklabels(labels_top, fontsize=8)
    ax4.set_xlabel('Cosine Similarity')
    ax4.set_title('4. Top-5 Ket qua tuong dong', fontsize=12, fontweight='bold')
    ax4.set_xlim(0, 1.1)
    ax4.invert_yaxis()
    for i, (bar, score) in enumerate(zip(bars, scores_top)):
        ax4.text(score + 0.01, i, f'{score:.4f}', va='center', fontsize=10, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'ket_qua_trung_gian.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n -> Đã lưu ảnh kết quả trung gian: {output_path}")

def search_similar_sounds(input_audio_path, top_k=5):
    if not os.path.exists(input_audio_path):
        print(f"Lỗi: Không tìm thấy file đầu vào '{input_audio_path}'")
        return

    print("=" * 60)
    print("HỆ THỐNG TÌM KIẾM TIẾNG ĐỘNG CƠ TƯƠNG ĐỒNG (CBAR)")
    print("=" * 60)
    
    # Bước 1: Trích xuất đặc trưng
    print("\n[Bước 1] Trích xuất đặc trưng (47 chiều) file đầu vào...")
    input_features = extract_features(input_audio_path)
    if input_features is None:
        return
        
    input_vector = np.array(input_features, dtype=float)
    
    # In ra vector đặc trưng (kết quả trung gian cho báo cáo)
    n = N_MFCC
    print(f"  -> Vector đặc trưng ({3*n+21} chiều):")
    print(f"     MFCC Mean     ({n}): {[round(x, 4) for x in input_features[:n]]}")
    print(f"     MFCC Std      ({n}): {[round(x, 4) for x in input_features[n:2*n]]}")
    print(f"     MFCC Autocorr ({n}): {[round(x, 4) for x in input_features[2*n:3*n]]}")
    print(f"     Spec Mean      (5): {[round(x, 4) for x in input_features[3*n:3*n+5]]}")
    print(f"     Spec Std       (5): {[round(x, 4) for x in input_features[3*n+5:3*n+10]]}")
    print(f"     Autocorr       (5): {[round(x, 4) for x in input_features[3*n+10:3*n+15]]}")
    print(f"     Energy Bands   (3): Low={input_features[3*n+15]:.4f}, Mid={input_features[3*n+16]:.4f}, High={input_features[3*n+17]:.4f}")
    print(f"     Tempo             : {input_features[3*n+18]:.2f} BPM")
    print(f"     Envelope Std      : {input_features[3*n+19]:.6f}")
    print(f"     Silence Ratio     : {input_features[3*n+20]:.6f}")
    
    # Bước 2: Đọc CSDL
    print(f"\n[Bước 2] Đọc dữ liệu từ CSDL ({DB_NAME})...")
    db_data = get_all_features_from_db()
    if not db_data:
        return
    print(f"  -> Đã nạp {len(db_data)} bản ghi từ CSDL.")
        
    # Bước 3: Chuẩn hóa và áp dụng trọng số
    db_vectors = np.array([item["features"] for item in db_data])
    
    print(f"\n[Bước 3] Chuẩn hóa StandardScaler + Weighted Feature Fusion (MFCC×0.6 / Spectral×0.4)...")
    scaler = StandardScaler()
    db_vectors_scaled = scaler.fit_transform(db_vectors)
    input_vector_scaled = scaler.transform([input_vector])[0]
    
    w_mfcc = 0.6
    w_spectral = 0.4
    mfcc_end = 3 * N_MFCC  # 60 chiều MFCC
    
    db_vectors_scaled[:, :mfcc_end]  *= w_mfcc
    db_vectors_scaled[:, mfcc_end:]  *= w_spectral
    
    input_vector_scaled[:mfcc_end]   *= w_mfcc
    input_vector_scaled[mfcc_end:]   *= w_spectral
    
    # Bước 4: Tính Cosine Similarity
    print(f"\n[Bước 4] Tính Cosine Similarity với {len(db_data)} files trong CSDL...")
    results = []
    
    for i, item in enumerate(db_data):
        db_vector = db_vectors_scaled[i]
        try:
            distance = cosine(input_vector_scaled, db_vector)
            similarity = 1 - distance
            similarity = max(0.0, min(1.0, similarity))
        except:
            similarity = 0.0
            
        results.append({
            "file_path": item["file_path"],
            "label": item["label"],
            "similarity": similarity
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
    
    # Bước 6: Sinh kết quả trung gian (ảnh)
    print(f"\n[Bước 6] Sinh kết quả trung gian (Waveform, Spectrogram, Feature Vector, Similarity Chart)...")
    try:
        generate_intermediate_results(input_audio_path, input_features, top_results)
    except Exception as e:
        print(f"  Cảnh báo: Không thể sinh ảnh kết quả trung gian: {e}")
    
    print("\n" + "=" * 60)
    print("HOÀN TẤT QUÁ TRÌNH TÌM KIẾM.")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Sử dụng: py search_engine.py <đường_dẫn_file_âm_thanh>")
    else:
        input_file = sys.argv[1]
        search_similar_sounds(input_file)
