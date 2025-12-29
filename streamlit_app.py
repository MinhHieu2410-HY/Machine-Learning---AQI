import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ================== CẤU HÌNH TRANG ==================
st.set_page_config(
    page_title="DỰ ĐOÁN AQI HÀ NỘI",
    page_icon="🌫️",
    layout="wide"  # Rộng hơn để sliders ở giữa đẹp
)

# ================== HEADER VỚI LOGO TRƯỜNG & MINH HỌA ==================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("🌫️ DỰ ĐOÁN MỨC ĐỘ Ô NHIỄM KHÔNG KHÍ - AQI")
    
st.markdown("---")

# ================== TẢI DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH ==================
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

with st.spinner("Đang tải dữ liệu và huấn luyện mô hình Random Forest..."):
    rf_model, feature_names = load_and_train_model()

# ================== ĐIỀU CHỈNH THÔNG SỐ MÔI TRƯỜNG Ở GIỮA TRANG ==================
st.markdown("### 🔧 Điều chỉnh thông số môi trường")

# Chia sliders thành 3 cột để gọn gàng và ở giữa
col1, col2, col3 = st.columns(3)

with col1:
    CO = st.slider("CO", 0.0, 1000.0, 150.0, step=10.0)
    NO2 = st.slider("NO₂", 0.0, 200.0, 40.0, step=5.0)
    O3 = st.slider("O₃", 0.0, 200.0, 30.0, step=5.0)
    PM10 = st.slider("PM10", 0.0, 600.0, 80.0, step=10.0)

with col2:
    PM25 = st.slider("PM2.5", 0.0, 500.0, 50.0, step=5.0)
    SO2 = st.slider("SO₂", 0.0, 100.0, 10.0, step=2.0)
    Clouds = st.slider("Mây che phủ (%)", 0, 100, 50)
    Precipitation = st.slider("Lượng mưa (mm)", 0.0, 20.0, 0.0, step=0.5)

with col3:
    Pressure = st.slider("Áp suất (hPa)", 990, 1030, 1010)
    Humidity = st.slider("Độ ẩm (%)", 30, 100, 70)
    Temperature = st.slider("Nhiệt độ (°C)", 10.0, 40.0, 25.0, step=0.5)
    WindSpeed = st.slider("Tốc độ gió (m/s)", 0.0, 10.0, 2.0, step=0.2)

# Tạo input dataframe
input_data = [CO, NO2, O3, PM10, PM25, SO2, Clouds, Precipitation, Pressure, Humidity, Temperature, WindSpeed]
input_df = pd.DataFrame([input_data], columns=feature_names)

# ================== DỰ ĐOÁN ==================
prediction = rf_model.predict(input_df)[0]

def get_aqi_info(aqi):
    if aqi <= 50: return "TỐT", "🟢", "#00e400", "Không khí trong lành! Ra ngoài thoải mái 🌳"
    elif aqi <= 100: return "TRUNG BÌNH", "🟡", "#ffff00", "Không khí bình thường. Người nhạy cảm chú ý."
    elif aqi <= 150: return "KÉM", "🟠", "#ff7e00", "Trẻ em, người già nên hạn chế ra ngoài lâu."
    elif aqi <= 200: return "XẤU", "🔴", "#ff0000", "Ô nhiễm nặng. Nên đeo khẩu trang N95!"
    elif aqi <= 300: return "RẤT XẤU", "🟣", "#8f3f97", "Cảnh báo sức khỏe nghiêm trọng!"
    else: return "NGUY HIỂM", "🟤", "#7e0023", "Ở nhà thôi! Đóng cửa, bật máy lọc không khí 😷🏠"

level, emoji, color, advice = get_aqi_info(prediction)

# ================== KẾT QUẢ DỰ ĐOÁN ==================
st.markdown("---")
st.markdown("### 📊 KẾT QUẢ DỰ ĐOÁN")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(f"<h1 style='text-align: center; color: {color};'>AQI: {prediction:.1f}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center;'>{emoji} {level}</h2>", unsafe_allow_html=True)
