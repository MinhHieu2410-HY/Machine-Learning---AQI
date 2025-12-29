# Machine-Learning---AQI
# Dự Án Machine Learning: Dự Đoán Mức Độ Ô Nhiễm Không Khí Tại Hà Nội

## Giới thiệu đề tài
- **Bài toán**: Dự đoán chỉ số chất lượng không khí (AQI) tại Hà Nội dựa trên dữ liệu ô nhiễm (PM2.5, PM10, CO, NO2, O3, SO2) và thời tiết (nhiệt độ, độ ẩm, gió, mưa,...). Đây là bài toán hồi quy (dự đoán giá trị liên tục) và phân loại (mức độ ô nhiễm).
- **Mục tiêu**: Xây dựng mô hình để cảnh báo sớm ô nhiễm, so sánh Linear Regression và Random Forest Regressor. Ý nghĩa thực tiễn: Bảo vệ sức khỏe cộng đồng tại Hà Nội, nơi ô nhiễm thường cao vào mùa đông.

## Dataset
- **Nguồn data**: Từ GitHub của namanhnt - [Hanoi-Air-Quality-Analysis](https://github.com/namanhnt/Hanoi-Air-Quality-Analysis/blob/main/Data/hanoi-aqi-weather-data.csv).
- **Link tải**: Tải file CSV từ link trên (khoảng 8785 mẫu, 18 cột).
- **Mô tả cột**:
  - UTC Time: Thời gian UTC.
  - AQI: Chỉ số AQI (biến mục tiêu).
  - CO, NO2, O3, PM10, PM25, SO2: Nồng độ chất ô nhiễm.
  - Clouds, Precipitation, Pressure, Relative Humidity, Temperature, UV Index, Wind Speed: Yếu tố thời tiết.
  - Các cột khác (City, Country Code, Timezone): Không dùng trong mô hình.
- Data mẫu nhỏ được đặt trong thư mục `data/` (data_sample.csv - 100 dòng đầu để demo).

## Pipeline
1. **Tiền xử lý**: Đọc data, loại bỏ cột không cần (City, Timezone, v.v.), xử lý missing values (nếu có), xử lý outlier bằng IQR, chia train/test (80/20 theo thời gian để tránh data leakage).
2. **Train**: Huấn luyện Linear Regression và Random Forest Regressor (với siêu tham số mặc định: 100 cây, không giới hạn độ sâu).
3. **Evaluate**: Sử dụng MAE, RMSE, R² cho hồi quy; Accuracy, F1-score, Confusion Matrix, AUC cho phân loại (chuyển AQI thành nhị phân: Good ≤100, Bad >100).
4. **Inference**: Dự đoán AQI mới dựa trên input đặc trưng.

## Mô hình sử dụng
- **Linear Regression**: Đơn giản, giả định tuyến tính. Lý do chọn: Baseline để so sánh, chạy nhanh.
- **Random Forest Regressor**: Ensemble phi tuyến, xử lý tốt mối quan hệ phức tạp (ví dụ: PM2.5 và AQI phi tuyến). Lý do chọn: Độ chính xác cao, chống overfit, phù hợp dữ liệu môi trường biến động.

## Kết quả
- **Linear Regression**:
  - MAE: ~20-25, RMSE: ~30-35, R²: ~0.80.
  - Phân loại nhị phân: Accuracy 80%, F1-score 0.78, AUC 0.50 (kém, không phân biệt tốt).
- **Random Forest**:
  - MAE: ~10-15, RMSE: ~15-20, R²: ~0.95.
  - Phân loại nhị phân: Accuracy 90%, F1-score 0.90, AUC 0.96 (xuất sắc).
- So sánh: Random Forest vượt trội, bắt được xu hướng phi tuyến (PM2.5 quyết định 95-98% AQI).
- Feature Importance (Random Forest): PM2.5 (cao nhất), PM10, NO2, Temperature, Wind Speed.

## Hướng dẫn chạy
### Cài môi trường
- Python 3.8+.
- Cài dependencies: `pip install -r requirements.txt`.

### Chạy train
- Chạy `python app/train.py` (huấn luyện và lưu model vào models/ - tạo thư mục nếu chưa có).

### Chạy demo/inference
- Truy cập streamlit.io
- Chọn My app > Create app > Deploy a public app from GitHub > Paste GitHub URL
- Gán link https://github.com/MinhHieu2410-HY/Machine-Learning---AQI/blob/main/demo/streamlit_app.py vô GitHub URL
- Click Deloy

## Cấu trúc thư mục dự án
- app/: Source code chính (preprocess.py, train.py, predict.py, utils.py).
- demo/: Notebook demo (demo.ipynb).
- data/: Data mẫu nhỏ và hướng dẫn tải full data.
- reports/: Báo cáo (Machine_Learning_Report.pdf).
- slides/: Slide thuyết trình (Machine_Learning_Slides.pdf).
- models/: Lưu model đã train (không commit lên GitHub, dùng .gitignore).
- requirements.txt
- README.md
- .gitignore

## Tác giả
- Họ tên: Nguyễn Minh Hiếu
- Mã SV: 12423049
- Lớp: 124231
- GV hướng dẫn: PGS. TS. Nguyễn Văn Hậu
