# BÁO CÁO ĐỒ ÁN: TÌM KIẾM ÂM THANH ĐỘNG CƠ (CBAR)

## 1. GIỚI THIỆU CHUNG
Hệ thống truy xuất âm thanh dựa trên nội dung (Content-Based Audio Retrieval - CBAR) thực hiện tìm kiếm các mẫu động cơ tương đồng thông qua phân tích đặc trưng vật lý của tín hiệu.

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python 3.10+ |
| Thư viện lõi | Librosa, Numpy, Scipy, Scikit-learn |
| Lưu trữ | SQLite 3 |

---

## 2. QUY TRÌNH TIỀN XỬ LÝ

### 2.1. Chuẩn hóa và Zero-padding
- **Resampling:** $f_s = 16{,}000$ Hz, kênh Mono.
- **Độ dài cố định:** 5.0 giây.
- **Vấn đề Zero-padding:** Để đồng bộ hóa dữ liệu phục vụ xử lý hàng loạt, kỹ thuật Zero-padding được áp dụng cho các file ngắn hơn 5s. Tuy nhiên, việc bù thêm các mẫu bằng 0 sẽ làm sai lệch các tham số thống kê (kéo thấp giá trị Mean, làm tăng Std nhân tạo).

### 2.2. Kỹ thuật Masking (Giải pháp tối ưu hóa)
Để loại bỏ ảnh hưởng tiêu cực của Zero-padding, hệ thống áp dụng kỹ thuật **Masking (Mặt nạ năng lượng)** trong quá trình trích xuất đặc trưng:
- Hệ thống tính toán năng lượng RMS cho từng khung hình ($32$ ms).
- Các khung có năng lượng thấp hơn ngưỡng $\epsilon$ được xác định là "khung im lặng" hoặc "phần bù zero".
- Toàn bộ các phép tính thống kê (Mean, Std, Autocorrelation) chỉ được thực hiện trên các khung hình có tín hiệu thực (Active Frames).

$$ \text{Active Frames} = \{ f : RMS(f) > \epsilon \} $$

Giải pháp này đảm bảo vector đặc trưng của một file âm thanh là **bất biến** bất kể nó được lưu trữ dưới độ dài gốc hay được padding thêm im lặng.

---

## 3. TRÍCH XUẤT ĐẶC TRƯNG (VECTOR 47 CHIỀU)

### 3.1. Nhóm MFCC (26 chiều)
- **MFCC Mean (13 chiều):** Đặc trưng âm sắc trung bình (tính trên Active Frames).
- **MFCC Std (13 chiều):** Độ lệch chuẩn âm sắc (tính trên Active Frames).

### 3.2. Nhóm Đặc trưng Phổ — Mean (5 chiều)
Giá trị trung bình của các đặc trưng vật lý tính trên Active Frames:
- **Năng lượng RMS**, **Zero Crossing Rate**, **Spectral Centroid**, **Spectral Bandwidth**, **Spectral Rolloff**.

### 3.3. Nhóm Đặc trưng Phổ — Std (5 chiều)
Độ lệch chuẩn của đặc trưng phổ qua các Active Frames, thể hiện mức độ ổn định của động cơ.

### 3.4. Tự tương quan bậc 1 — Autocorrelation (5 chiều)
Đo tính chu kỳ và cấu trúc thời gian của tín hiệu động cơ:
$$r_1 = \frac{\sum_{t=1}^{T-1}(x_t - \bar{x})(x_{t+1} - \bar{x})}{\sum_{t=1}^{T}(x_t - \bar{x})^2}$$

### 3.5. Phân bố năng lượng theo dải tần (3 chiều)
Tính tỷ lệ năng lượng trên 3 dải: Low (0-2kHz), Mid (2-4kHz), High (4-8kHz).

### 3.6. Các đặc trưng bổ trợ (3 chiều)
- **Tempo:** Nhịp điệu chu kỳ nổ (BPM).
- **Envelope Std:** Độ biến thiên đường bao biên độ.
- **Silence Ratio:** Tỷ lệ khoảng lặng tự nhiên của động cơ.

---

## 4. THUẬT TOÁN TÌM KIẾM (SEARCH ENGINE)

### 4.1. Chuẩn hóa và Trọng số
Vector 47 chiều được chuẩn hóa bằng StandardScaler:
$$z = \frac{x - \mu}{\sigma}$$

Áp dụng trọng số: $w_{mfcc} = 0.6$, $w_{spectral} = 0.4$.

### 4.2. Cosine Similarity
$$S_C(v_q, v_i) = \frac{v_q \cdot v_i}{\|v_q\| \cdot \|v_i\|}$$

---

## 5. KẾT QUẢ VÀ ĐÁNH GIÁ
Hệ thống trả về Top-5 kết quả. Việc sử dụng kỹ thuật Masking giúp cải thiện đáng kể độ chính xác khi tìm kiếm các đoạn âm thanh có độ dài khác nhau, loại bỏ hiện tượng sai lệch vị trí trong không gian vector do padding.

---

## PHỤ LỤC: CẤU TRÚC MÃ NGUỒN
- `dataset_builder.py`: Chuẩn hóa 16kHz, Mono, 5s (có padding).
- `feature_extractor.py`: Trích xuất 47 đặc trưng với **thuật toán Masking**.
- `database.py`: SQLite + lưu vector JSON.
- `search_engine.py`: Engine tìm kiếm: StandardScaler + Weighted Fusion + Cosine.
