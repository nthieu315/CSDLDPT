import librosa
import numpy as np

# Ngưỡng biên độ để xác định "khoảng lặng" (silence)
SILENCE_THRESHOLD = 0.02
# Ngưỡng năng lượng khung để loại bỏ khung im lặng (tránh ảnh hưởng của Zero-padding)
FRAME_ENERGY_THRESHOLD = 0.005

def lag1_autocorrelation(x):
    """Tính tự tương quan bậc 1 của chuỗi giá trị."""
    if len(x) < 2 or np.std(x) == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])

def extract_features(file_path):
    """
    Trích xuất 47 đặc trưng âm thanh với kỹ thuật MASKING 
    để loại bỏ ảnh hưởng của Zero-padding.
    """
    try:
        # Tải audio với sr mặc định
        y, sr = librosa.load(file_path, sr=None)
        
        # Cấu hình Framing & Windowing
        N_FFT = 512
        HOP_LENGTH = 160
        
        # 1. Trích xuất các đặc trưng theo từng khung (Frame-level)
        # ------------------------------------------------------
        # mfccs: (n_mfcc, n_frames)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP_LENGTH)
        # rms_f: (1, n_frames)
        rms_f = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
        
        # Lấy mask từ năng lượng RMS (chuyển về 1D: n_frames)
        active_mask = rms_f[0] > FRAME_ENERGY_THRESHOLD
        
        # Nếu file toàn im lặng, ta lấy toàn bộ (tránh lỗi)
        if not np.any(active_mask):
            active_mask = np.ones_like(rms_f[0], dtype=bool)
            
        # 2. Lọc các khung hình "sống" cho tất cả đặc trưng
        # ------------------------------------------------------
        # Lọc MFCC
        mfccs_active = mfccs[:, active_mask]
        
        # Trích xuất các đặc trưng khác và lọc ngay
        centroid_f = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        bandwidth_f = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        rolloff_f = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        zcr_f = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
        
        centroid_active = centroid_f[active_mask]
        bandwidth_active = bandwidth_f[active_mask]
        rolloff_active = rolloff_f[active_mask]
        zcr_active = zcr_f[active_mask]
        rms_active = rms_f[0][active_mask]
        
        # 3. Tính toán thống kê trên các khung ĐÃ LỌC
        # ------------------------------------------------------
        # Nhóm MFCC (26 chiều)
        mfccs_mean = np.mean(mfccs_active, axis=1)
        mfccs_std = np.std(mfccs_active, axis=1)
        
        # Nhóm Spectral Mean (5 chiều)
        spec_means = [
            np.mean(centroid_active), np.mean(bandwidth_active),
            np.mean(rolloff_active), np.mean(zcr_active), np.mean(rms_active)
        ]
        
        # Nhóm Spectral Std (5 chiều)
        spec_stds = [
            np.std(centroid_active), np.std(bandwidth_active),
            np.std(rolloff_active), np.std(zcr_active), np.std(rms_active)
        ]
        
        # Nhóm Autocorrelation (5 chiều)
        spec_acorrs = [
            lag1_autocorrelation(centroid_active), lag1_autocorrelation(bandwidth_active),
            lag1_autocorrelation(rolloff_active), lag1_autocorrelation(zcr_active),
            lag1_autocorrelation(rms_active)
        ]
        
        # 4. Các đặc trưng khác
        # ------------------------------------------------------
        # Năng lượng dải tần (Dùng phổ gốc y)
        S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
        power = S ** 2
        total_energy = np.sum(power) + 1e-10
        low_energy = np.sum(power[freqs < 2000]) / total_energy
        mid_energy = np.sum(power[(freqs >= 2000) & (freqs < 4000)]) / total_energy
        high_energy = np.sum(power[freqs >= 4000]) / total_energy
        
        # Tempo và Envelope
        tempo = librosa.beat.beat_track(y=y, sr=sr)[0]
        if hasattr(tempo, '__len__'): tempo = float(tempo[0])
        envelope_std = float(np.std(rms_active))
        
        # Silence Ratio: Đo trên toàn bộ file (bao gồm cả padding) để phản ánh tính chất file
        silence_ratio = np.sum(np.abs(y) < SILENCE_THRESHOLD) / len(y)
        
        # Ghép Vector 47 chiều
        feature_vector = np.concatenate((
            mfccs_mean, mfccs_std,
            spec_means, spec_stds, spec_acorrs,
            [low_energy, mid_energy, high_energy],
            [tempo, envelope_std, silence_ratio]
        ))
        
        return [float(x) for x in feature_vector]
        
    except Exception as e:
        print(f"Lỗi trích xuất đặc trưng file {file_path}: {e}")
        return None
