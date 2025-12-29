import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ================== CẤU HÌNH TRANG ==================
st.set_page_config(page_title="DỰ ĐOÁN AQI HÀ NỘI - NGUYỄN MINH HIỂU", 
                   page_icon="🌫️", 
                   layout="centered")

# ================== TRANG BÌA BÀI TẬP LỚN ==================
st.markdown("""
<style>
    .big-title {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
    }
    .medium-title {
        font-size: 24px;
        text-align: center;
    }
    .info {
        font-size: 18px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.image("https://dean1665.vn/uploads/school/kthy.jpg", width=200)  # Logo trường chính thức

st.markdown("<div class='big-title'>BỘ GIÁO DỤC VÀ ĐÀO TẠO</div>", unsafe_allow_html=True)
st.markdown("<div class='big-title'>TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT HƯNG YÊN</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<div class='big-title'>BÀI TẬP LỚN</div>", unsafe_allow_html=True)
st.markdown("<div class='medium-title'>DỰ ĐOÁN MỨC ĐỘ Ô NHIỄM KHÔNG KHÍ</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<div class='info'>NGÀNH: CÔNG NGHỆ THÔNG TIN</div>", unsafe_allow_html=True)
st.markdown("<div class='info'>CHUYÊN NGÀNH: KHOA HỌC MÁY TÍNH</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<div class='info'>SINH VIÊN: NGUYỄN MINH HIỂU</div>", unsafe_allow_html=True)
st.markdown("<div class='info'>MÃ SINH VIÊN: 12423049</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<div class='info'>MÃ LỚP: 124231</div>", unsafe_allow_html=True)
st.markdown("<div class='info'>GV HƯỚNG DẪN: PGS. TS. NGUYỄN VĂN HẬU</div>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("<div class='medium-title'>HƯNG YÊN – 2025</div>", unsafe_allow_html=True)

st.markdown("---")

# Nút để chuyển sang phần demo dự đoán
if st.button("👉 Bắt đầu sử dụng ứng dụng dự đoán AQI"):
    st.experimental_rerun()

# ================== PHẦN DỰ ĐOÁN AQI ==================
st.title("🌫️ Dự Đoán Chỉ Số Chất Lượng Không Khí (AQI) Hà Nội")
st.markdown("**Mô hình: Random Forest Regressor** | Dữ liệu thực tế Hà Nội 2023-2024")
st.markdown("**Sinh viên thực hiện: Nguyễn Minh Hiếu - 12423049**")
st.markdown("---")

# Tải dữ liệu + train model
@st.cache_resource
def load_and_train_model():
    url = "https://github.com/namanhnt/Hanoi-Air-Quality-Analysis/raw/main/Data/hanoi-aqi-weather-data.csv"
    df = pd.read_csv(url)
    
    df = df.drop(['UTC Time', 'City', 'Country Code', 'Timezone', 'UV Index'], axis=1, errors='ignore')
    
    features = ['CO', 'NO2', 'O3', 'PM10', 'PM25', 'SO2', 
                'Clouds', 'Precipitation', 'Pressure', 
                'Relative Humidity', 'Temperature', 'Wind Speed']
    X = df[features]
    y = df['AQI']
    
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, features

with st.spinner("Đang huấn luyện mô hình Random Forest..."):
    rf_model, feature_names = load_and_train_model()

st.success("✅ Mô hình sẵn sàng! Điều chỉnh các chỉ số bên trái để dự đoán.")

# Sidebar sliders
st.sidebar.header("🔧 Điều chỉnh chỉ số môi trường")

def get_user_input():
    CO = st.sidebar.slider("CO", 0.0, 1000.0, 150.0, step=10.0)
    NO2 = st.sidebar.slider("NO₂", 0.0, 200.0, 40.0, step=5.0)
    O3 = st.sidebar.slider("O₃", 0.0, 200.0, 30.0, step=5.0)
    PM10 = st.sidebar.slider("PM10", 0.0, 600.0, 80.0, step=10.0)
    PM25 = st.sidebar.slider("PM2.5 ★ (yếu tố chính)", 0.0, 500.0, 50.0, step=5.0)
    SO2 = st.sidebar.slider("SO₂", 0.0, 100.0, 10.0, step=2.0)
    Clouds = st.sidebar.slider("Mây che phủ (%)", 0, 100, 50)
    Precipitation = st.sidebar.slider("Mưa (mm)", 0.0, 20.0, 0.0, step=0.5)
    Pressure = st.sidebar.slider("Áp suất (hPa)", 990, 1030, 1010)
    Humidity = st.sidebar.slider("Độ ẩm (%)", 30, 100, 70)
    Temperature = st.sidebar.slider("Nhiệt độ (°C)", 10.0, 40.0, 25.0, step=0.5)
    WindSpeed = st.sidebar.slider("Tốc độ gió (m/s)", 0.0, 10.0, 2.0, step=0.2)
    
    data = [CO, NO2, O3, PM10, PM25, SO2, Clouds, Precipitation, Pressure, Humidity, Temperature, WindSpeed]
    return pd.DataFrame([data], columns=feature_names)

input_df = get_user_input()

# Dự đoán
prediction = rf_model.predict(input_df)[0]

# Mức AQI + màu
def get_aqi_info(aqi):
    if aqi <= 50: return "TỐT", "🟢", "#00e400", "Không khí trong lành! Ra ngoài thoải mái 🌳"
    elif aqi <= 100: return "TRUNG BÌNH", "🟡", "#ffff00", "Bình thường Hà Nội. Người nhạy cảm chú ý."
    elif aqi <= 150: return "KÉM", "🟠", "#ff7e00", "Trẻ em, người già hạn chế ra ngoài lâu."
    elif aqi <= 200: return "XẤU", "🔴", "#ff0000", "Ô nhiễm nặng. Đeo khẩu trang N95!"
    elif aqi <= 300: return "RẤT XẤU", "🟣", "#8f3f97", "Cảnh báo sức khỏe nghiêm trọng!"
    else: return "NGUY HIỂM", "🟤", "#7e0023", "Ở nhà thôi! Đóng cửa, bật máy lọc 😷🏠"

level, emoji, color, advice = get_aqi_info(prediction)

# Hiển thị kết quả
st.markdown("---")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown(f"<h1 style='text-align: center; color: {color};'>AQI: {prediction:.1f}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center;'>{emoji} {level}</h2>", unsafe_allow_html=True)

st.markdown(f"**💡 Lời khuyên:** {advice}")

st.markdown("---")
st.caption("🔥 Tip: Kéo **PM2.5** lên 250+ để thấy AQI vọt lên mức NGUY HIỂM như mùa đông Hà Nội thực tế!")
st.caption("Bài tập lớn Machine Learning - Trường ĐH Sư phạm Kỹ thuật Hưng Yên - 2025")
