# Machine-Learning---AQI
# Dự Án Machine Learning: Dự Đoán Mức Độ Ô Nhiễm Không Khí Tại Hà Nội

## Giới thiệu đề tài
- **Bài toán**: Dự đoán chỉ số chất lượng không khí (AQI) tại Hà Nội dựa trên dữ liệu ô nhiễm (PM2.5, PM10, CO, NO2, O3, SO2) và thời tiết (nhiệt độ, độ ẩm, gió, mưa,...). Đây là bài toán hồi quy (dự đoán giá trị liên tục) và phân loại (mức độ ô nhiễm).
- **Mục tiêu**: Xây dựng mô hình để cảnh báo sớm ô nhiễm, so sánh Linear Regression và Random Forest Regressor. Ý nghĩa thực tiễn: Bảo vệ sức khỏe cộng đồng, nơi ô nhiễm thường cao.

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
1. **Tiền xử lý**: Đọc data, loại bỏ cột không cần (City, Timezone, ...), xử lý missing values, xử lý outlier bằng IQR, chia train/test (80/20 theo thời gian).
2. **Train**: Huấn luyện Linear Regression và Random Forest Regressor (với siêu tham số mặc định: 100 cây, không giới hạn độ sâu).
3. **Evaluate**: Sử dụng Accuracy, F1-score, Confusion Matrix, AUC.
4. **Inference**: Dự đoán AQI mới dựa trên input đặc trưng.

## Mô hình sử dụng
- **Linear Regression**: Đơn giản, giả định tuyến tính.
  Lý do chọn: Baseline để so sánh, chạy nhanh.
- **Random Forest Regressor**: Ensemble phi tuyến, xử lý tốt mối quan hệ phức tạp.
  Lý do chọn: Độ chính xác cao, chống overfit, phù hợp dữ liệu môi trường biến động.

## Kết quả
- **Linear Regression**:
  - MAE: ~20-25, RMSE: ~30-35, R²: ~0.80.
  - Phân loại nhị phân: Accuracy 80%, F1-score 0.78, AUC 0.50 (kém, không phân biệt tốt).
- **Random Forest**:
  - MAE: ~10-15, RMSE: ~15-20, R²: ~0.95.
  - Phân loại nhị phân: Accuracy 90%, F1-score 0.90, AUC 0.96 (tốt).
- So sánh: Random Forest vượt trội, bắt được xu hướng phi tuyến.

## Hướng dẫn chạy
### Cài môi trường
- Python 3.11.
- Cài dependencies: `pip install -r requirements.txt`.

### Chạy train
1. Chạy trên Google Colab:
- Truy cập https://colab.research.google.com
- Chọn File > Upload notebook (hoặc Import > Upload), sau đó chọn file Machine_Learning.ipynb trên máy tính.
- Chọn menu Runtime > Run all (Ctrl + F9).
Lưu ý:
- Cần tải file dataset hanoi-aqi-weather-data.csv trên Google Drive (đặt trong thư mục /MyDrive/Colab Notebooks/).
- Nếu file dataset nằm ở vị trí khác, hãy sửa dòng code:
  df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/hanoi-aqi-weather-data.csv')
  thành đường dẫn đã lưu file.
- Nếu không muốn dùng Drive, có thể upload file CSV trực tiếp lên Colab (dùng biểu tượng thư mục bên trái > Upload) rồi sửa đường dẫn thành "/content/hanoi-aqi-weather-data.csv."

2. Chạy trên VS Code (máy tính cá nhân):
- Cài đặt và mở VS Code
- Cài extension Jupyter và Python từ Marketplace.
- Mở file Machine_Learning.ipynb bằng VS Code.
- Tạo hoặc chọn môi trường Python (dùng virtual environment).
- Cài các thư viện cần thiết bằng lệnh trong terminal:
  pip install pandas numpy matplotlib seaborn scikit-learn
- Tải file dataset hanoi-aqi-weather-data.csv về máy.
- Sửa dòng đọc file thành đường dẫn cục bộ vị trí lưu file code, ví dụ:
  df = pd.read_csv('C:/Users/YourName/Documents/hanoi-aqi-weather-data.csv')
- Chạy từng cell bằng nút Run Cell hoặc Shift + Enter, hoặc chạy toàn bộ bằng Run All.

3. Chạy trên Jupyter Notebook/JupyterLab:
- Cài đặt Python.
- Mở terminal/command prompt, cài Jupyter:
  pip install jupyter hoặc pip install jupyterlab.
- Khởi động:
  jupyter notebook hoặc jupyter lab
- Chọn và mở file Machine_Learning.ipynb.
- Cài thư viện:
  pip install pandas numpy matplotlib seaborn scikit-learn
- Tải dataset về máy và sửa đường dẫn đọc file thành đường dẫn cục bộ (giống như VS Code).
- Chạy từng cell hoặc chọn Cell > Run All.

### Chạy demo/inference
- Truy cập streamlit.io
- Chọn My app > Create app > Deploy a public app from GitHub > Paste GitHub URL
- Gán link https://github.com/MinhHieu2410-HY/Machine-Learning---AQI/blob/main/demo/streamlit_app.py vô GitHub URL
- Click Deloy

## Cấu trúc thư mục dự án
- app/: Source code chính.
- demo/: streamlit_app.py.
- data/: Data mẫu nhỏ.
- reports/: Báo cáo.
- slides/: Slide thuyết trình.
- requirements.txt
- README.md
- .gitignore
- runtime.txt.

## Tác giả
- Họ tên: Nguyễn Minh Hiếu
- Mã SV: 12423049
- Lớp: 124231
- GV hướng dẫn: PGS. TS. Nguyễn Văn Hậu
