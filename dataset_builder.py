import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import librosa
import soundfile as sf
import numpy as np

# Cấu hình thư mục
RAW_DIR = 'dataset/data'  # Thư mục chứa các file âm thanh gốc của bạn
OUT_DIR = 'dataset_normalized'    # Thư mục chứa các file đã chuẩn hóa

# Cấu hình chuẩn hóa theo yêu cầu đề bài
TARGET_SR = 16000
TARGET_DURATION = 5.0 # giây
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)

def process_audio(file_path, output_path):
    """
    Đọc, chuẩn hóa (Mono, 16kHz, 5s) và lưu file âm thanh.
    """
    try:
        # 1. Đọc file, librosa mặc định gom kênh (mono=True) và chỉ định sr=TARGET_SR
        # Khi resample về 16kHz, librosa tự động áp dụng Anti-aliasing low-pass filter (chặn các tần số > 8kHz)
        y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
        
        # 2. Chuẩn hóa mức khuếch đại (Amplitude Normalization)
        # Đưa biên độ tín hiệu về khoảng [-1.0, 1.0] để đồng nhất âm lượng
        y = librosa.util.normalize(y)
        
        # 3. Xử lý độ dài: Cắt nếu dài hơn 5s, Pad (chèn 0) nếu ngắn hơn 5s
        if len(y) > TARGET_SAMPLES:
            # File dài -> cắt
            y = y[:TARGET_SAMPLES]
        elif len(y) < TARGET_SAMPLES:
            # File ngắn -> padding
            padding_length = TARGET_SAMPLES - len(y)
            y = np.pad(y, (0, padding_length), mode='constant')
            
        # 4. Lưu lại ra định dạng WAV chuẩn
        sf.write(output_path, y, TARGET_SR, subtype='PCM_16')
        return True
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {e}")
        return False

def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        
    if not os.path.exists(RAW_DIR):
        print(f"Không tìm thấy thư mục '{RAW_DIR}'. Vui lòng tạo và để các file gốc vào đó.")
        return

    # Quét toàn bộ file trong RAW_DIR
    valid_exts = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    count = 0
    
    print("Bắt đầu quá trình chuẩn hóa dữ liệu...")
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_exts:
                input_path = os.path.join(root, file)
                
                # Lấy tên thư mục chứa file làm label (ví dụ: Airplane, Cars...)
                label = os.path.basename(root).replace(" ", "_").replace(",", "")
                
                # Tạo tên file đầu ra duy nhất
                # Thay thế các ký tự không hợp lệ bằng underscore nếu cần
                output_filename = f"{label}_{count:04d}.wav"
                output_path = os.path.join(OUT_DIR, output_filename)
                
                print(f"Đang xử lý [{count+1}]: {file} -> {output_filename}")
                success = process_audio(input_path, output_path)
                
                if success:
                    count += 1
                    
    print(f"\nHoàn thành! Đã chuẩn hóa {count} files. Kết quả lưu trong thư mục '{OUT_DIR}'.")

if __name__ == "__main__":
    main()
