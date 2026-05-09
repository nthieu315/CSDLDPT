# BÁO CÁO ĐỒ ÁN: TÌM KIẾM ÂM THANH ĐỘNG CƠ (CBAR) - PHIÊN BẢN TỐI ƯU 81 CHIỀU

## 1. GIỚI THIỆU CHUNG
Hệ thống truy xuất âm thanh dựa trên nội dung (Content-Based Audio Retrieval - CBAR) thực hiện tìm kiếm các mẫu động cơ tương đồng (ô tô, máy bay, trực thăng) thông qua phân tích các đặc trưng vật lý và thống kê của tín hiệu âm thanh.

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python 3.10+ |
| Thư viện lõi | Librosa, Numpy, Scipy, Scikit-learn |
| Lưu trữ | SQLite 3 |

---

## 2. QUY TRÌNH TIỀN XỬ LÝ (PREPROCESSING)

### 2.1. Cắt khoảng lặng (Silence Trimming)
Hệ thống áp dụng `librosa.effects.trim` với ngưỡng `top_db=30` để loại bỏ các đoạn im lặng không mang thông tin ở đầu và cuối file. Điều này đảm bảo:
- Vector đặc trưng tập trung hoàn toàn vào đoạn âm thanh thực tế.
- Tránh sai lệch chỉ số `Silence Ratio` do các đoạn im lặng kéo dài không liên quan.

### 2.2. Chuẩn hóa và Masking
- **Resampling:** Tín hiệu được đưa về cùng tần số lấy mẫu (mặc định của librosa hoặc sr gốc).
- **Masking (Mặt nạ năng lượng):** 
    - Hệ thống tính toán năng lượng RMS cho từng khung hình.
    - Chỉ các khung hình có năng lượng vượt ngưỡng (`active_mask`) mới được đưa vào tính toán thống kê (Mean, Std, Autocorrelation).
    - Công thức: $$ \text{Active Frames} = \{ f : RMS(f) > \text{threshold} \} $$
    - Kỹ thuật này giúp loại bỏ ảnh hưởng của Zero-padding và nhiễu nền cực thấp.

---

## 3. TRÍCH XUẤT ĐẶC TRƯNG (VECTOR 81 CHIỀU)
Hệ thống sử dụng vector **81 chiều** để mô tả chi tiết âm sắc, phổ và nhịp điệu của động cơ.

### 3.1. Chi tiết các nhóm đặc trưng

| STT | Nhóm đặc trưng | Số chiều | Ý nghĩa / Vai trò |
|:---:|---|:---:|---|
| **1-20** | **MFCC Mean** | 20 | Đặc trưng âm sắc trung bình (hệ số cepstral). |
| **21-40** | **MFCC Std** | 20 | Độ biến thiên của âm sắc, thể hiện sự ổn định/gầm rú. |
| **41-60** | **MFCC Autocorr** | 20 | Tính chu kỳ bậc 1 của âm sắc (Lag-1 autocorrelation). |
| **61-65** | **Spectral Mean** | 5 | Trung bình: Centroid, Bandwidth, Rolloff, ZCR, RMS. |
| **66-70** | **Spectral Std** | 5 | Độ lệch chuẩn của các thông số phổ trên. |
| **71-75** | **Spectral Autocorr**| 5 | Tính chu kỳ của các thông số phổ. |
| **76-78** | **Energy Bands** | 3 | Tỷ lệ năng lượng dải Bass (0-2k), Mid (2-4k), Treble (>4k). |
| **79** | **Tempo** | 1 | Nhịp điệu chu kỳ nổ (BPM). |
| **80** | **Envelope Std** | 1 | Biến thiên độ lớn biên độ, phân biệt động cơ có tiếng nổ đanh gọn hay kéo dài. |
| **81** | **Silence Ratio** | 1 | Tỷ lệ khoảng lặng tự nhiên trong đoạn âm thanh thực. |

### 3.2. Ý nghĩa vật lý cụ thể

- **20 MFCC (Hệ số Cepstral):** Việc tăng lên 20 hệ số (so với 13 mặc định) giúp nắm bắt chi tiết hơn các "vân âm thanh" ở tần số trung-cao, cực kỳ quan trọng để phân biệt tiếng máy bay (vòng quay cao) và ô tô.
- **Spectral Centroid:** Xác định "độ sáng" của âm thanh. Động cơ nhỏ/RPM cao có Centroid cao hơn động cơ diesel nặng.
- **Energy Bands:** 
    - *Low (0-2kHz):* Năng lượng từ chu kỳ piston.
    - *Mid (2-4kHz):* Tiếng ma sát cơ khí, tiếng van nạp/xả.
    - *High (>4kHz):* Tiếng rít gió, tiếng van hoặc nhiễu cao tần.
- **Lag-1 Autocorrelation:** Đo lường sự tương quan giữa các khung hình liên tiếp. Giá trị cao thể hiện âm thanh động cơ đều đặn, ổn định.

---

## 4. THUẬT TOÁN TÌM KIẾM (SEARCH ENGINE)

### 4.1. Chuẩn hóa và Trọng số
Vector 81 chiều được chuẩn hóa bằng `StandardScaler` dựa trên toàn bộ dữ liệu trong CSDL để đưa các đặc trưng về cùng thang đo.
Sau đó, hệ thống áp dụng **Weighted Feature Fusion**:
- **Trọng số MFCC ($w=0.6$):** Ưu tiên nhận diện âm sắc đặc trưng.
- **Trọng số Spectral ($w=0.4$):** Cân đối với các đặc tính vật lý và năng lượng.

### 4.2. Cosine Similarity
Độ tương đồng được tính bằng công thức Cosine:
$$S_C(v_q, v_i) = \frac{v_q \cdot v_i}{\|v_q\| \cdot \|v_i\|}$$
Giá trị trả về nằm trong khoảng $[0, 1]$, càng gần 1 càng tương đồng cao.

---

## 5. PHỤ LỤC: CẤU TRÚC MÃ NGUỒN
- `feature_extractor.py`: Trích xuất 81 chiều, áp dụng **Trim** và **Masking**.
- `database.py`: Quản lý SQLite, tự động rebuild DB khi thay đổi thuật toán.
- `search_engine.py`: Xử lý logic tìm kiếm, chuẩn hóa và xuất kết quả trung gian.
