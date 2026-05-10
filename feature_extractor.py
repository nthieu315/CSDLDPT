import librosa
import numpy as np
from scipy.signal import find_peaks

# Ngưỡng biên độ để xác định "khoảng lặng" (silence)
SILENCE_THRESHOLD = 0.02
# Ngưỡng năng lượng khung để loại bỏ khung im lặng (tránh ảnh hưởng của Zero-padding)
FRAME_ENERGY_THRESHOLD = 0.005

def lag1_autocorrelation(x):
    """Tính tự tương quan bậc 1 của chuỗi giá trị."""
    if len(x) < 2 or np.std(x) == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])

N_MFCC = 20

def extract_features(file_path):
    """
    Trích xuất 88 đặc trưng âm thanh theo 10 nhóm phân loại:
    [0:3]   Nhóm 1 - Tần số    : ZCR (Mean, Std, Autocorr)
    [3:6]   Nhóm 2 - Biên độ   : RMS (Mean, Std, Autocorr)
    [6:8]   Nhóm 3 - Thời gian : Tempo, Silence Ratio
    [8:75]  Nhóm 4 - Phổ âm    : MFCC×60 + Bandwidth×3 + Rolloff×3 + Contrast×1 = 67
    [75:77] Nhóm 5 - Hình dạng : Envelope Std, Pct Above Threshold
    [77:79] Nhóm 6 - Độ phức tạp: Num Peaks, Num Valleys
    [79:82] Nhóm 7 - Biên độ màu: Energy Bands Low/Mid/High
    [82:86] Nhóm 8 - Độ chói   : Centroid×3 + Spectral Flatness×1
    [86]    Nhóm 9 - Độ chạy   : Attack Time
    [87]    Nhóm 10- Độ suy giảm: Decay Time
    """
    try:
        # Tải audio với sr mặc định
        y, sr = librosa.load(file_path, sr=None)
        
        # Cắt bỏ silence ở đầu và cuối file (trim)
        # top_db=30: frame nào thấp hơn max_amplitude - 30dB được coi là silence
        y, _ = librosa.effects.trim(y, top_db=30)
        
        # Cấu hình Framing & Windowing
        N_FFT = 512
        HOP_LENGTH = 160
        
        # 1. Trích xuất các đặc trưng theo từng khung (Frame-level)
        # ------------------------------------------------------
        # mfccs: (n_mfcc, n_frames)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
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
        # MFCC (20×3 = 60 chiều)
        mfccs_mean  = np.mean(mfccs_active, axis=1)
        mfccs_std   = np.std(mfccs_active, axis=1)
        mfccs_acorr = np.array([lag1_autocorrelation(mfccs_active[i]) for i in range(N_MFCC)])

        # Thống kê riêng từng đặc trưng phổ (mean, std, autocorr)
        def get_stats(arr):
            return float(np.mean(arr)), float(np.std(arr)), lag1_autocorrelation(arr)

        zcr_mean, zcr_std, zcr_acorr = get_stats(zcr_active)
        rms_mean, rms_std, rms_acorr = get_stats(rms_active)
        bw_mean, bw_std, bw_acorr    = get_stats(bandwidth_active)
        ro_mean, ro_std, ro_acorr    = get_stats(rolloff_active)
        ct_mean, ct_std, ct_acorr    = get_stats(centroid_active)
        
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
        
        # Tempo & Envelope Std
        tempo = librosa.beat.beat_track(y=y, sr=sr)[0]
        if hasattr(tempo, '__len__'): tempo = float(tempo[0])
        envelope_std = rms_std  # Đã tính ở trên (= std của rms_active)
        
        # Silence Ratio: Đo trên toàn bộ file (bao gồm cả padding) để phản ánh tính chất file
        silence_ratio = np.sum(np.abs(y) < SILENCE_THRESHOLD) / len(y)

        # -----------------------------------------------------------------------
        # 5. Các đặc trưng bổ sung (indices 81-87)
        # -----------------------------------------------------------------------

        # [81] Spectral Flatness: Đo mức độ đồng đều của phổ âm
        # Giá trị gần 1 → nhiễu trắng; gần 0 → âm có cấu trúc hài âm rõ ràng
        flatness_f = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        spectral_flatness_mean = float(np.mean(flatness_f[active_mask]))

        # [82] Spectral Contrast: Độ chênh lệch đỉnh – thung lũng năng lượng phổ
        # n_bands=3 → giá trị mean trên toàn bộ băng tần
        contrast_f = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_bands=3
        )  # shape: (n_bands+1, n_frames) = (4, n_frames)
        spectral_contrast_mean = float(np.mean(contrast_f[:, active_mask]))

        # [83-84] Num Peaks & Num Valleys: Định lượng độ phức tạp của waveform
        # Dùng scipy.signal.find_peaks trên tín hiệu thô đã trim
        peaks, _   = find_peaks(y,  height=SILENCE_THRESHOLD)
        valleys, _ = find_peaks(-y, height=SILENCE_THRESHOLD)
        # Chuẩn hoá theo độ dài tín hiệu để không phụ thuộc vào duration
        num_peaks   = len(peaks)   / len(y)
        num_valleys = len(valleys) / len(y)

        # [85] Percentage Above Threshold: Phần trăm thời gian biên độ > 0.5
        AMPLITUDE_THRESHOLD = 0.5
        pct_above_threshold = float(np.sum(np.abs(y) > AMPLITUDE_THRESHOLD) / len(y))

        # [86-87] Attack Time & Decay Time
        # Attack: Thời gian từ đầu đến đỉnh biên độ (RMS envelope)
        # Decay : Thời gian từ đỉnh đến khi RMS giảm xuống còn 1/e (~36.8%) của đỉnh
        rms_env   = rms_f[0]                            # shape: (n_frames,)
        peak_idx  = int(np.argmax(rms_env))
        peak_val  = rms_env[peak_idx]
        # Attack: số frame tới đỉnh × thời gian mỗi frame
        attack_time = float(peak_idx * HOP_LENGTH / sr)
        # Decay: tìm frame sau đỉnh mà RMS giảm xuống dưới peak/e
        decay_threshold = peak_val / np.e
        after_peak = rms_env[peak_idx:]
        decay_frames = np.where(after_peak < decay_threshold)[0]
        if len(decay_frames) > 0:
            decay_time = float(decay_frames[0] * HOP_LENGTH / sr)
        else:
            decay_time = float((len(rms_env) - peak_idx) * HOP_LENGTH / sr)
        
        # Ghép Vector 88 chiều theo 10 nhóm phân loại
        # [0:3]   Nhóm 1 - Tần số    : ZCR Mean/Std/Autocorr
        # [3:6]   Nhóm 2 - Biên độ   : RMS Mean/Std/Autocorr
        # [6:8]   Nhóm 3 - Thời gian : Tempo, Silence Ratio
        # [8:68]  Nhóm 4a- Phổ âm    : MFCC Mean(20)+Std(20)+Autocorr(20)
        # [68:71] Nhóm 4b- Phổ âm    : Bandwidth Mean/Std/Autocorr
        # [71:74] Nhóm 4c- Phổ âm    : Rolloff Mean/Std/Autocorr
        # [74:75] Nhóm 4d- Phổ âm    : Spectral Contrast Mean
        # [75:77] Nhóm 5 - Hình dạng : Envelope Std, Pct Above Threshold
        # [77:79] Nhóm 6 - Độ phức tạp: Num Peaks, Num Valleys
        # [79:82] Nhóm 7 - Biên độ màu: Energy Low/Mid/High
        # [82:85] Nhóm 8a- Độ chói   : Centroid Mean/Std/Autocorr
        # [85:86] Nhóm 8b- Độ chói   : Spectral Flatness
        # [86]    Nhóm 9 - Độ chạy   : Attack Time
        # [87]    Nhóm 10- Độ suy giảm: Decay Time
        feature_vector = np.concatenate((
            [zcr_mean, zcr_std, zcr_acorr],                        # [0:3]   Nhóm 1
            [rms_mean, rms_std, rms_acorr],                        # [3:6]   Nhóm 2
            [tempo, silence_ratio],                                # [6:8]   Nhóm 3
            mfccs_mean, mfccs_std, mfccs_acorr,                   # [8:68]  Nhóm 4a
            [bw_mean, bw_std, bw_acorr],                           # [68:71] Nhóm 4b
            [ro_mean, ro_std, ro_acorr],                           # [71:74] Nhóm 4c
            [spectral_contrast_mean],                              # [74:75] Nhóm 4d
            [envelope_std, pct_above_threshold],                   # [75:77] Nhóm 5
            [num_peaks, num_valleys],                              # [77:79] Nhóm 6
            [low_energy, mid_energy, high_energy],                 # [79:82] Nhóm 7
            [ct_mean, ct_std, ct_acorr, spectral_flatness_mean],   # [82:86] Nhóm 8
            [attack_time],                                         # [86]    Nhóm 9
            [decay_time]                                           # [87]    Nhóm 10
        ))
        
        return [float(x) for x in feature_vector]
        
    except Exception as e:
        print(f"Lỗi trích xuất đặc trưng file {file_path}: {e}")
        return None
