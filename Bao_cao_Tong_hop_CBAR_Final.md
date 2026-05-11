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

Các bước tiền xử lý được thiết kế để định hình không gian đặc trưng chính xác nhất, loại bỏ nhiễu và tập trung vào bản chất cơ khí:

### 2.1. Cắt khoảng lặng (Silence Trimming)
- **Lý do sử dụng**: Âm thanh động cơ thường có đoạn "chờ" hoặc nhiễu nền ở đầu/cuối file. Việc cắt khoảng lặng đảm bảo các đặc trưng thống kê (như trung bình năng lượng) phản ánh đúng hoạt động của động cơ thay vì bị kéo thấp bởi các đoạn im lặng vô ích.
- **Cơ chế**: Sử dụng ngưỡng `top_db=30` để xác định ranh giới tín hiệu.

### 2.2. Phân khung và Cửa sổ hóa (Framing & Windowing)
- **Lý do sử dụng**: Tín hiệu âm thanh là tín hiệu phi dừng. Việc chia nhỏ thành các khung cực ngắn (20-30ms) cho phép coi tín hiệu là dừng trong khoảng đó để áp dụng các phép biến đổi Fourier (FFT). Hàm cửa sổ giúp làm mịn biên khung hình, giảm thiểu sai số phổ.
- **Thông số**: `N_FFT=512`, `Hop_Length=160`.

### 2.3. Lọc năng lượng khung (Frame Energy Masking)
- **Lý do sử dụng**: Trong tiếng động cơ thực tế luôn có các khoảng nghỉ cơ khí siêu ngắn hoặc tiếng gió nhiễu cực thấp. Việc lọc bỏ các khung có năng lượng RMS < `0.005` giúp hệ thống chỉ tính toán trên các đoạn có "tiếng nổ" thực sự, làm tăng độ nhạy bén của vector đặc trưng.

---

## 3. PHÂN TÍCH ĐẶC TRƯNG ÂM THANH (FEATURE ANALYSIS)

Quá trình phân tích đặc trưng được thực hiện theo hai giai đoạn: Xác định các miền lý thuyết chính và chia nhỏ thành các nhóm thực nghiệm để gán trọng số.

### 3.1. Phân loại theo Miền Lý thuyết (3 Miền chính)

Dựa trên lý thuyết xử lý tín hiệu số (DSP), các đặc trưng âm thanh được phân loại vào 3 miền nền tảng:

#### A. Miền Thời gian (Time-domain Features)
Phân tích trực tiếp trên biên độ sóng âm $x[t]$.
- **RMS (Root Mean Square)**: Đo năng lượng. Công thức: $RMS = \sqrt{\frac{1}{N} \sum x[n]^2}$.
- **ZCR (Zero Crossing Rate)**: Đo tốc độ đổi dấu tín hiệu. Công thức: $ZCR = \frac{1}{2(N-1)} \sum |\text{sgn}(x[n]) - \text{sgn}(x[n-1])|$.
- **Waveform Shape**: Mô tả hình dạng đường bao và độ phức tạp (Peaks/Valleys).

#### B. Miền Tần số (Frequency-domain Features)
Chuyển đổi tín hiệu sang miền tần số bằng phép biến đổi Fourier nhanh (FFT).
- **MFCCs**: Mô phỏng thính giác người, tách biệt đặc trưng của buồng đốt và thân máy.
- **Spectral Centroid/Flatness**: Đo trọng tâm năng lượng (độ sáng) và tính hài âm của âm thanh.

#### C. Miền Động lực học & Nhịp điệu (Temporal/Rhythmic)
Mô tả sự biến đổi theo chu kỳ dài.
- **Tempo**: Nhịp điệu vòng tua máy (RPM).
- **ADSR Envelope**: Attack/Decay Time phản ánh quy luật đánh lửa và xả thải cơ khí.

---

### 3.2. Triển khai Thực nghiệm (Nguồn gốc Vector 88 chiều)

Để đảm bảo tính bền vững, hệ thống trích xuất tổ hợp các chỉ số: **Trung bình (Mean)**, **Độ lệch chuẩn (Std)** và **Tự tương quan (Autocorr)** của các đặc trưng gốc.

| STT | Nhóm triển khai | Số chiều | Cách cấu thành 88 chiều |
|:---:|---|:---:|---|
| **1** | **Tần số (ZCR)** | 3 | Mean, Std, Autocorr (1x3) |
| **2** | **Biên độ (RMS)** | 3 | Mean, Std, Autocorr (1x3) |
| **3** | **Thời gian (Tempo)** | 2 | Tempo (1), Silence Ratio (1) |
| **4** | **Phổ âm (MFCC)** | 67 | **MFCC (60)**: 20 hệ số x 3 chỉ số (M,S,A); **Bandwidth (3)**: M,S,A; **Rolloff (3)**: M,S,A; **Contrast (1)**: Mean |
| **5** | **Hình dạng** | 2 | Envelope Std (1), Pct Above Threshold (1) |
| **6** | **Độ phức tạp** | 2 | Num Peaks (1), Num Valleys (1) |
| **7** | **Biên độ màu** | 3 | Energy Bands (Low, Mid, High) |
| **8** | **Độ chói (Centroid)** | 4 | **Centroid (3)**: M,S,A; **Flatness (1)**: Mean |
| **9** | **Độ chạy (Attack)** | 1 | Attack Time (1) |
| **10**| **Độ suy giảm (Decay)** | 1 | Decay Time (1) |
| | **TỔNG CỘNG** | **88** | |

---

## 4. THUẬT TOÁN TÌM KIẾM VÀ TÍNH TƯƠNG ĐỒNG (SEARCH ENGINE)

### 4.1. Chuẩn hóa StandardScaler (Z-score)
Đưa tất cả các chiều đặc trưng về cùng một mặt bằng thống kê để tránh việc các đặc trưng có thang đo lớn áp đảo các đặc trưng nhỏ.

### 4.2. Group-wise Weighted Cosine Similarity
Thuật toán tính độ tương đồng theo từng nhóm thuộc tính và tổng hợp có trọng số:
$$Sim_{final} = \sum_{i=1}^{10} (w_i \times \text{CosineSim}(Group_i))$$

**Bảng trọng số tối ưu:** Phổ âm (40%), Độ chói (15%), ADSR (15%), Biên độ màu (10%), Khác (20%).

---

## 5. ĐÁNH GIÁ HIỆU SUẤT THỰC NGHIỆM (EVALUATION)

Trên tập dữ liệu thực tế **1,974 mẫu**, hệ thống đạt được:
- **Accuracy (Top-1)**: **91.79%**
- **mAP**: **0.6095**
- **Precision@5**: **88.27%**

**Phân tích nhầm lẫn**: Nhầm lẫn chủ yếu giữa Airplane và Supercars (22 trường hợp) do âm rít ở vòng tua cao có phổ âm tương đồng.

---

## 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
Hệ thống chứng minh tính hiệu quả vượt trội trong việc phân tích âm thanh cơ khí. Các hướng phát triển tiếp theo bao gồm áp dụng **K-means Clustering** để phân mảnh trạng thái và **Relevance Feedback** để điều chỉnh trọng số theo ý muốn người dùng.
