# BÁO CÁO ĐỒ ÁN: HỆ THỐNG TÌM KIẾM ÂM THANH ĐỘNG CƠ (CBAR) - VECTOR 88 CHIỀU

---

## 1. GIỚI THIỆU CHUNG

Hệ thống truy xuất âm thanh dựa trên nội dung (**Content-Based Audio Retrieval - CBAR**) thực hiện tìm kiếm các mẫu âm thanh động cơ tương đồng (ô tô, máy bay, trực thăng) thông qua phân tích đặc trưng vật lý và thống kê của tín hiệu âm thanh, không dựa vào nhãn hay siêu dữ liệu.

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python 3.10+ |
| Thư viện lõi | Librosa, NumPy, SciPy, Scikit-learn |
| Lưu trữ | SQLite 3 |
| File chính | `feature_extractor.py`, `database.py`, `search_engine.py` |

---

## 2. QUY TRÌNH TIỀN XỬ LÝ (PREPROCESSING)

Trước khi trích xuất đặc trưng, tín hiệu âm thanh thô được xử lý qua hai bước:

### 2.1. Cắt khoảng lặng (Silence Trimming)
Hệ thống áp dụng `librosa.effects.trim` với ngưỡng `top_db=30` để loại bỏ các đoạn im lặng ở đầu và cuối file. Đảm bảo vector đặc trưng chỉ tập trung vào âm thanh thực tế của động cơ và tránh sai lệch thông số.

### 2.2. Masking theo năng lượng khung (Frame Energy Masking)
Sau khi trim, tín hiệu được phân tách thành các khung (`N_FFT=512`, `HOP_LENGTH=160`). Chỉ các khung có năng lượng RMS vượt ngưỡng (`0.005`) mới tham gia tính toán thống kê. Kỹ thuật này loại bỏ ảnh hưởng của Zero-padding và nhiễu nền cực thấp.

---

## 3. TRÍCH XUẤT ĐẶC TRƯNG (VECTOR 88 CHIỀU)

Hệ thống sử dụng vector **88 chiều** được phân chia thành **10 nhóm thuộc tính** rõ ràng để đánh giá toàn diện tính chất âm thanh.

### BẢNG PHÂN LOẠI VECTOR ĐẶC TRƯNG 88 CHIỀU

| STT | Nhóm thuộc tính | Số chiều | Thành phần chi tiết từ vector 88 chiều |
|:---:|---|:---:|---|
| **1** | **Tần số (Frequency)** | 3 | ZCR (Mean, Std, Autocorr bậc 1) |
| **2** | **Biên độ (Amplitude)** | 3 | RMS / Energy (Mean, Std, Autocorr) |
| **3** | **Thời gian (Time)** | 2 | Tempo (Nhịp điệu) và Silence Ratio (Tỷ lệ khoảng lặng) |
| **4** | **Phổ âm (Spectrum)** | 67 | 60 chiều MFCCs, Bandwidth (3), Rolloff (3), và Spectral Contrast (1) |
| **5** | **Hình dạng (Waveform)** | 2 | Envelope Std và Percentage Above Threshold (Phần trăm trên ngưỡng) |
| **6** | **Độ phức tạp (Complexity)**| 2 | Num Peaks (Số lượng đỉnh) và Num Valleys (Số lượng hố) |
| **7** | **Biên độ màu (Timbre)** | 3 | Energy Bands: Low, Mid, High (Mô tả "màu sắc" âm thanh) |
| **8** | **Độ chói (Brightness)** | 4 | Spectral Centroid (3 chiều) và Spectral Flatness (1 chiều) |
| **9** | **Độ chạy (Attack)** | 1 | Attack Time (Thời gian từ khi bắt đầu đến khi đạt cường độ cực đại) |
| **10** | **Độ suy giảm (Decay)** | 1 | Decay Time (Thời gian âm thanh giảm dần từ đỉnh) |
| | **TỔNG CỘNG** | **88** | |

---

## 4. THUẬT TOÁN TÌM KIẾM VÀ TÍNH TƯƠNG ĐỒNG (SEARCH ENGINE)

### 4.1. Chuẩn hóa từng phần (StandardScaler)
Vector 88 chiều của toàn bộ CSDL được chuẩn hóa bằng `StandardScaler` (Z-score normalization). Quá trình chuẩn hóa đảm bảo mọi giá trị đặc trưng dù có thang đo khác nhau đều được đưa về cùng một mặt bằng (trung bình 0, phương sai 1) để không làm sai lệch thuật toán đo khoảng cách.

### 4.2. Cosine Similarity theo Nhóm (Group-wise Cosine Similarity)
Thay vì tính toán khoảng cách trên toàn bộ vector 88 chiều cùng lúc (điều này có thể dẫn đến việc nhóm Phổ âm 67 chiều hoàn toàn áp đảo các nhóm khác), hệ thống **tính toán Cosine Similarity riêng biệt cho từng nhóm trong 10 nhóm thuộc tính**.

Mỗi khoảng cách nhóm ($Sim_{Group\_i}$) trả về kết quả nằm trong khoảng $[0, 1]$.

### 4.3. Bảng phân bổ trọng số

| Nhóm thuộc tính | Trọng số | Lý do và Ý nghĩa kỹ thuật |
|---|:---:|---|
| **Nhóm 4: Phổ âm** | 40% | Chứa 60 chiều MFCCs. Tài liệu xác nhận đây là bộ mô tả hiệu quả nhất để bắt được "vân âm sắc" (timbre) và cấu trúc hài âm đặc trưng của động cơ [129, History]. |
| **Nhóm 8: Độ chói** | 15% | Kết hợp Centroid và Spectral Flatness. Đây là nhóm "chí mạng" để phân biệt các loại máy có RPM cao (sáng) với động cơ Diesel nặng (trầm), hoặc trực thăng với ô tô [76, 77, History]. |
| **Nhóm 9 & 10: Độ chạy & Độ suy giảm** | 15% | Mô tả chu kỳ ADSR của nhịp nổ piston. Rất quan trọng để phân biệt tính chất cơ khí đanh gọn hay kéo dài của tiếng máy. |
| **Nhóm 7: Biên độ màu** | 10% | Sử dụng Energy Bands để phân loại nguồn năng lượng theo dải tần (piston vs cơ khí vs gió), giúp xác định bản chất của động cơ. |
| **Nhóm 3 & 6: Thời gian & Độ phức tạp** | 10% | Nhịp điệu (Tempo) và cấu trúc đỉnh/hố giúp lọc nhiễu môi trường và bắt được tính chu kỳ ổn định của động cơ hoạt động bình thường. |
| **Nhóm 1, 2 & 5: Tần số, Biên độ & Hình dạng** | 10% | Các đặc trưng vật lý bậc thấp cơ bản. Trọng số thấp để tránh việc các thay đổi về độ to (Loudness) do khoảng cách thu âm làm nhiễu kết quả tìm kiếm. |

### 4.4. Công thức tổng hợp (Weighted Final Similarity)

Độ tương đồng cuối cùng ($Sim_{final}$) được tính theo công thức:

$$Sim_{final} = \sum_{i=1}^{10} (w_i \times Sim_{Group\_i})$$

Trong đó:
- $w_i$ là trọng số của nhóm thứ $i$ (tổng các trọng số bằng 1.0).
- $Sim_{Group\_i}$ là độ tương đồng Cosine của nhóm thuộc tính đó.

Việc tính toán theo công thức này đảm bảo **nhóm có số chiều lớn (như Phổ âm) không áp đảo các nhóm có số chiều ít nhưng quan trọng (như Độ chạy/Độ suy giảm)**.

### 4.5. Tinh chỉnh qua Phản hồi (Relevance Feedback)
Trọng số của 10 nhóm hoàn toàn có thể được điều chỉnh dựa trên đánh giá thực tế của người dùng. Ví dụ: Nếu hệ thống nhầm lẫn giữa hai loại động cơ có nhịp nổ khác nhau nhưng âm sắc giống nhau, người vận hành có thể tăng trọng số cho nhóm **Độ chạy (Attack)** và **Thời gian (Time)** để gia tăng độ nhạy bén của hệ thống.

---

## 5. CẤU TRÚC LƯU TRỮ VÀ MÃ NGUỒN

### Cấu trúc file hệ thống:
- `feature_extractor.py`: Trích xuất vector 88 chiều chia theo 10 nhóm. Đã được dọn dẹp mã nguồn để tối ưu luồng tính toán thống kê.
- `database.py`: Lưu trữ vector vào CSDL SQLite (`engine_sounds.db`).
- `search_engine.py`: Chịu trách nhiệm Load CSDL, Chuẩn hóa (Scaler), Cắt vector thành 10 nhóm nhỏ, Tính Group Cosine Similarity và Tổng hợp trọng số $Sim_{final}$.

### Lưu ý về cấu trúc Tìm kiếm CSDL:
Trong tương lai, với cấu trúc 88 chiều và việc chia trọng số tính toán theo nhóm, hệ thống nên cân nhắc việc lưu trữ từng nhóm vector này trong các cột (column) riêng biệt của cơ sở dữ liệu thay vì một mảng JSON duy nhất. Việc này sẽ giúp quá trình load dữ liệu, tính toán ma trận và gán trọng số linh hoạt hơn ở cấp độ Database (như sử dụng Vector DB chuyên dụng).
