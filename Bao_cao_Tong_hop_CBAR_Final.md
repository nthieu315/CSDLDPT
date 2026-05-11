# BÁO CÁO TỔNG HỢP TOÀN DIỆN VÀ CHI TIẾT: HỆ THỐNG TRUY XUẤT ÂM THANH ĐỘNG CƠ (CBAR)
## PHÂN TÍCH VECTOR 88 CHIỀU, CƠ SỞ TOÁN HỌC VÀ THỰC NGHIỆM HIỆU SUẤT

---

## 1. GIỚI THIỆU CHUNG (INTRODUCTION)
Hệ thống Truy xuất Âm thanh dựa trên Nội dung (**Content-Based Audio Retrieval - CBAR**) là một giải pháp công nghệ nhằm tìm kiếm các mẫu âm thanh động cơ tương đồng thông qua việc phân tích đặc trưng vật lý và thống kê của tín hiệu. Thay vì dựa vào nhãn (metadata), hệ thống "nghe" và "hiểu" âm thanh thông qua các đại diện toán học đa chiều.

| Thành phần | Công nghệ triển khai |
|---|---|
| **Ngôn ngữ** | Python 3.10+ |
| **Thư viện lõi** | Librosa (Xử lý âm thanh), NumPy (Toán học), Scikit-learn (Chuẩn hóa/Đánh giá) |
| **Lưu trữ** | SQLite 3 (Lưu trữ vector đặc trưng) |
| **Mục tiêu** | Phân loại và tìm kiếm Máy bay, Ô tô, Siêu xe, Trực thăng, Máy cưa |

---

## 2. QUY TRÌNH TIỀN XỬ LÝ CHUYÊN SÂU (PREPROCESSING)

### 2.1. Cắt khoảng lặng (Silence Trimming)
- **Công thức**: $y_{trimmed} = \{y[t] \mid |y[t]| > \epsilon\}$
- **Ý nghĩa**: Loại bỏ các đoạn nhiễu hoặc khoảng lặng ở đầu/cuối file thu âm. Việc này đảm bảo các đặc trưng thống kê như trung bình năng lượng không bị kéo thấp bởi các đoạn không có tín hiệu. Hệ thống sử dụng ngưỡng `top_db=30`.

### 2.2. Phân khung và Cửa sổ hóa (Framing & Windowing)
- **Công thức**: $x_w[n] = x[n] \cdot w[n]$
- **Lý thuyết**: Âm thanh là tín hiệu biến thiên. Để phân tích, ta chia tín hiệu thành các khung cực ngắn (20-30ms) để coi nó là dừng (stationary). 
- **Thông số**: `N_FFT=512`, `Hop_Length=160`. Sử dụng hàm cửa sổ để tránh hiện tượng rò rỉ phổ (spectral leakage) khi thực hiện biến đổi Fourier.

### 2.3. Lọc năng lượng khung (Frame Energy Masking)
- **Cơ chế**: Chỉ các khung có năng lượng RMS > `0.005` mới được đưa vào tính toán đặc trưng.
- **Ý nghĩa thực tế**: Loại bỏ các "khoảng nghỉ" cơ khí giữa các nhịp nổ động cơ hoặc tiếng gió nhiễu nhẹ, giúp vector đặc trưng phản ánh chính xác bản chất của tiếng máy.

---

## 3. HỆ THỐNG ĐẶC TRƯNG VECTOR 88 CHIỀU (FEATURE ENGINEERING)

Hệ thống sử dụng vector **88 chiều** được chia thành 10 nhóm thuộc tính để đánh giá toàn diện tính chất âm thanh:

### BẢNG PHÂN LOẠI VÀ Ý NGHĨA VẬT LÝ

| STT | Nhóm thuộc tính | Số chiều | Ý nghĩa vật lý và Rationale |
|:---:|---|:---:|---|
| **1** | **Tần số (Frequency)** | 3 | ZCR (Tốc độ đổi dấu). Phân biệt tiếng rít cao (máy cưa) và tiếng trầm (xe tải). |
| **2** | **Biên độ (Amplitude)** | 3 | RMS (Năng lượng). Phản ánh cường độ hoạt động của động cơ. |
| **3** | **Thời gian (Time)** | 2 | Tempo (Nhịp điệu BPM) và Silence Ratio. Bắt nhịp nổ tuần hoàn của máy. |
| **4** | **Phổ âm (Spectrum)** | 67 | 60 chiều MFCCs. Đây là "vân tay âm thanh", mô tả cộng hưởng trong buồng đốt. |
| **5** | **Hình dạng (Waveform)** | 2 | Envelope Std. Đo mức độ biến động của đường bao biên độ. |
| **6** | **Độ phức tạp** | 2 | Số lượng Đỉnh/Thung lũng. Phân biệt âm thanh "mịn" hay "gai góc". |
| **7** | **Biên độ màu (Timbre)**| 3 | Năng lượng dải Low/Mid/High. Xác định cân bằng âm sắc (Bassy vs Trebly). |
| **8** | **Độ chói (Brightness)**| 4 | Spectral Centroid. Phân biệt động cơ tua máy cao (Sáng) vs máy Diesel (Trầm). |
| **9** | **Độ chạy (Attack)** | 1 | Attack Time. Tiếng nổ piston (nhanh) vs Tiếng gió phản lực (chậm). |
| **10** | **Độ suy giảm (Decay)**| 1 | Decay Time. Thời gian âm thanh tan biến sau mỗi chu kỳ nổ. |

---

## 4. THUẬT TOÁN TÌM KIẾM VÀ TÍNH TƯƠNG ĐỒNG (SEARCH ENGINE)

### 4.1. Chuẩn hóa StandardScaler (Z-score)
- **Công thức**: $z = (x - \mu) / \sigma$
- **Lý thuyết**: Đưa tất cả các chiều đặc trưng (dù là Hz, BPM hay biên độ) về cùng một mặt bằng thống kê. Nếu không có bước này, các đặc trưng có giá trị lớn sẽ làm lu mờ các đặc trưng nhỏ nhưng quan trọng.

### 4.2. Group-wise Weighted Cosine Similarity
Đây là thuật toán cốt lõi của hệ thống, cho phép so khớp dựa trên từng nhóm thuộc tính thay vì tính gộp toàn bộ vector:
$$Sim_{final} = \sum_{i=1}^{10} (w_i \times \text{CosineSim}(Group_i))$$

**Bảng phân bổ trọng số tối ưu:**
- **Nhóm 4 - Phổ âm (40%)**: Đặc trưng quan trọng nhất để nhận diện danh tính nguồn âm.
- **Nhóm 8 - Độ chói (15%)**: Chìa khóa để phân biệt trực thăng vs máy bay phản lực.
- **Nhóm 9 & 10 - ADSR (15%)**: Bắt được chu kỳ cơ khí của piston.
- **Nhóm 7 - Timbre (10%)**: Phân loại theo dải tần năng lượng.
- **Các nhóm khác (20%)**: Đảm bảo tính bền vững trước nhiễu môi trường.

---

## 5. ĐÁNH GIÁ HIỆU SUẤT THỰC NGHIỆM (EVALUATION)

Trên tập dữ liệu thực tế **1,974 mẫu**, hệ thống đạt được các chỉ số:

| Chỉ số | Kết quả | Ý nghĩa |
|---|---|---|
| **Accuracy (Precision@1)** | **91.79%** | Tỉ lệ file đứng đầu kết quả tìm kiếm là đúng nhãn. |
| **mAP (Mean Average Precision)**| **0.6095** | Khả năng xếp hạng các kết quả liên quan ở vị trí cao. |
| **Precision@5** | **88.27%** | Độ chính xác trong 5 kết quả trả về đầu tiên. |

### Phân tích Ma trận nhầm lẫn:
- **Ưu điểm**: Chainsaw (0.96) và Airplane (0.95) đạt độ chính xác cực cao do đặc trưng rất tách biệt.
- **Thách thức**: Nhầm lẫn giữa Airplane và Supercars (22 trường hợp) do âm rít ở vòng tua cao có phổ âm tương đồng.
- **Cars**: Hoạt động ổn định (90% precision), ít bị nhầm lẫn nhờ dải tần trung tính.

---

## 6. CẤU TRÚC HỆ THỐNG VÀ MÃ NGUỒN (SOURCE CODE)

- `feature_extractor.py`: Trích xuất vector 88 chiều chia theo 10 nhóm.
- `database.py`: Quản lý CSDL SQLite (`engine_sounds.db`).
- `search_engine.py`: Thực hiện Load CSDL, Chuẩn hóa và tính toán tương đồng trọng số.
- `evaluate_performance.py`: Script tự động đánh giá các chỉ số Precision, Recall và mAP.

---

## 7. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
Hệ thống CBAR 88 chiều cung cấp một giải pháp mạnh mẽ và có khả năng giải thích cao cho việc tìm kiếm âm thanh động cơ. 

**Hướng phát triển đề xuất:**
1.  **Relevance Feedback**: Cho phép người dùng điều chỉnh trọng số Tempo và Attack để tách biệt nhịp nổ piston.
2.  **K-means Clustering**: Phân mảnh các nhãn lớn thành các trạng thái (cất cánh, tăng tốc) để tăng độ mịn của tìm kiếm.
3.  **Indexing**: Sử dụng cấu trúc cây cho các bộ dữ liệu lớn hơn 100,000 mẫu.
