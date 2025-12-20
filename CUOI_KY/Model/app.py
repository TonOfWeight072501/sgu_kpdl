import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# 1. CẤU HÌNH TRANG
st.set_page_config(
    page_title="Dự báo Nhân sự (HR Analytics)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. LOAD CÁC FILE CẤU HÌNH (MODEL, SCALER, COLUMNS)
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('best_ensemble_model.pkl')
        scaler = joblib.load('scaler.pkl')
        model_cols = joblib.load('model_columns.pkl')
        return model, scaler, model_cols
    except FileNotFoundError as e:
        st.error(f"⚠️ Thiếu file quan trọng: {e}. Vui lòng upload đầy đủ: best_ensemble_model.pkl, scaler.pkl, model_columns.pkl")
        return None, None, None

model, scaler_obj, model_cols = load_assets()

# 3. XỬ LÝ SCALER THÔNG MINH
# Tạo từ điển mapping {tên_cột: (min, max)} từ scaler.pkl để tra cứu nhanh
# Cách này giúp app chạy đúng ngay cả khi scaler chứa nhiều cột thừa (như Gender,...)
scale_map = {}
if scaler_obj:
    try:
        # Kiểm tra xem scaler có lưu tên cột không (sklearn > 1.0)
        if hasattr(scaler_obj, 'feature_names_in_'):
            for name, min_val, max_val in zip(scaler_obj.feature_names_in_, scaler_obj.data_min_, scaler_obj.data_max_):
                scale_map[name] = {'min': min_val, 'max': max_val}
        else:
            st.warning("⚠️ File scaler.pkl quá cũ (không chứa tên cột). App sẽ dùng cấu hình mặc định.")
    except Exception as e:
        st.error(f"Lỗi khi đọc scaler: {e}")

# Hàm chuẩn hóa dùng giá trị thực tế từ scaler
def smart_normalize(value, col_name):
    if col_name in scale_map:
        cfg = scale_map[col_name]
        # Công thức Min-Max: (X - min) / (max - min)
        if cfg['max'] - cfg['min'] == 0: return 0
        return (value - cfg['min']) / (cfg['max'] - cfg['min'])
    # Fallback nếu không tìm thấy cột trong scaler (ít xảy ra)
    return value 

# Mapping tần suất (Frequency Encoding) - Cần khớp với lúc train
# Nếu trong scaler có cột 'JobTitle_Freq', giá trị min/max sẽ tự động được lấy
JOB_TITLE_FREQ = {
    'Teacher': 0.317, 'Accountant': 0.097, 'Sales': 0.05,
    'Engineer': 0.04, 'Nurse': 0.03, 'Manager': 0.02, 'Khác': 0.01
}

# 4. GIAO DIỆN NHẬP LIỆU
st.title("📊 Ứng dụng Dự báo Nghỉ việc & Sức khỏe Nhân sự")
st.markdown("---")

with st.form("hr_form"):
    col1, col2, col3 = st.columns(3)

    # --- Cột 1: Thông tin cá nhân ---
    with col1:
        st.subheader("1. Thông tin cá nhân")
        age = st.slider("Tuổi (Age)", 18, 60, 30)
        
        degree_map = {"Trung cấp/Cao đẳng": 0, "Cử nhân (Bachelor)": 1, "Thạc sĩ (Master)": 2, "Tiến sĩ (PhD)": 3}
        academic_degree = degree_map[st.selectbox("Bằng cấp", list(degree_map.keys()), index=1)]
        
        job_title = st.selectbox("Chức danh", list(JOB_TITLE_FREQ.keys()))
        
        salary = st.number_input("Lương tháng", 2000, 100000, 10000, step=500)
        allowances = st.number_input("Phụ cấp", 0, 2,1)

    # --- Cột 2: Kinh nghiệm & Công việc ---
    with col2:
        st.subheader("2. Kinh nghiệm & Công việc")
        years_exp = st.slider("Tổng năm kinh nghiệm", 0, 40, 5)
        years_last_org = st.slider("Số năm làm công ty cũ", 0, 40, 3) # Để tính Stability Ratio
        
        training_prog = st.slider("Số CT đào tạo (3 năm qua)", 0, 20, 2)
        
        promotion = 1 if st.radio("Được thăng chức xứng đáng?", ["Có", "Không"], horizontal=True) == "Có" else 0
        job_opps = 1 if st.radio("Có cơ hội việc làm khác?", ["Có", "Không"], horizontal=True) == "Có" else 0
        
        dist_map = {"Gần": 0, "Trung bình": 1, "Xa": 2}
        distance = dist_map[st.selectbox("Khoảng cách đi làm", list(dist_map.keys()))]

    # --- Cột 3: Tâm lý & Môi trường ---
    with col3:
        st.subheader("3. Tâm lý & Môi trường")
        
        level_map = {"Thấp/Kém": 0, "Trung bình": 1, "Cao/Tốt": 2}
        sat_map = {"Không hài lòng": 0, "Hài lòng": 1, "Rất hài lòng": 2}
        
        job_sat = sat_map[st.selectbox("Hài lòng công việc", list(sat_map.keys()), index=1)]
        env_sat = level_map[st.selectbox("Hài lòng môi trường", list(level_map.keys()), index=1)]
        job_support = level_map[st.selectbox("Sự hỗ trợ (Job Support)", list(level_map.keys()), index=1)]
        emo_commit = level_map[st.selectbox("Cam kết cảm xúc", list(level_map.keys()), index=1)]
        
        diff_map = {"Dễ dàng": 0, "Trung bình": 1, "Khó khăn": 2}
        job_engage = diff_map[st.selectbox("Mức độ gắn kết", list(diff_map.keys()), index=1)]
        wlb = diff_map[st.selectbox("Work-Life Balance", list(diff_map.keys()), index=1)]
        
        freq_map = {"Không": 0, "Thỉnh thoảng": 1, "Có/Thường xuyên": 2}
        phys_stress = freq_map[st.selectbox("Căng thẳng thể chất", list(freq_map.keys()))]
        psy_exhaustion = freq_map[st.selectbox("Kiệt quệ tâm lý", list(freq_map.keys()))]
        
        job_stability = 1 if st.radio("Công việc ổn định?", ["Có", "Không"], horizontal=True) == "Có" else 0

    submit = st.form_submit_button("🚀 Phân tích ngay")

# 5. XỬ LÝ & DỰ BÁO
if submit and model and model_cols:
    # A. Feature Engineering (Tính toán đặc trưng phái sinh)
    stability_ratio = years_last_org / years_exp if years_exp > 0 else 0.0
    
    # B. Tạo Dictionary dữ liệu thô (Raw Data)
    # Tên key phải khớp với (hoặc map được sang) tên trong scaler/model
    raw_input = {
        'Age': age,
        'Academic_degree': academic_degree,
        'Years_Experience': years_exp,
        'MonthlySalary': salary,
        'Allowances': allowances,
        'Get_ Deserved_Promotion': promotion,
        'Training_programs_ During_last_three_years': training_prog,
        'Job_Support': job_support,
        'Emotional_Commitment': emo_commit,
        'Job_Engagement': job_engage,
        'Distance_to_work': distance,
        'Work_Live_Balance': wlb,
        'Physical_Stress': phys_stress,
        'Psychological_Exhaustion': psy_exhaustion,
        'Job_Stability': job_stability,
        'Environment_Satisfaction': env_sat,
        'Job_Satisfaction': job_sat,
        'Job_Opportunities': job_opps,
        'JobTitle_Freq': JOB_TITLE_FREQ.get(job_title, 0.01),
        'Stability_Ratio': stability_ratio
    }

    # C. Chuẩn hóa dữ liệu (Scaling)
    # Chỉ lấy đúng các cột mà Model yêu cầu (dựa trên model_columns.pkl)
    input_ready = {}
    for col in model_cols:
        if col in raw_input:
            val = raw_input[col]
            # Nếu cột này có trong scaler, thì chuẩn hóa
            # Các cột binary (0/1) thường min=0, max=1 nên normalize vẫn đúng là chính nó
            input_ready[col] = smart_normalize(val, col)
        else:
            st.warning(f"Thiếu thông tin cho cột: {col}. Điền giá trị 0.")
            input_ready[col] = 0

    # D. Tạo DataFrame đúng thứ tự
    df_input = pd.DataFrame([input_ready])
    # Đảm bảo thứ tự cột tuyệt đối chính xác
    df_input = df_input[model_cols]

    # E. Dự báo
    try:
        proba = model.predict_proba(df_input)[0][1] # Xác suất lớp 1 (Nghỉ việc)
        
        st.markdown("---")
        st.subheader("📋 Kết quả phân tích")
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.metric("Nguy cơ nghỉ việc", f"{proba*100:.1f}%")
            if proba > 0.5:
                st.error("🔴 Nguy cơ CAO")
            else:
                st.success("🟢 Nguy cơ THẤP")
        
        with col_res2:
            st.progress(proba)
            if proba > 0.5:
                st.write("**Khuyến nghị:** Cần gặp gỡ nhân viên để trao đổi về lộ trình thăng tiến và cân bằng công việc.")
            else:
                st.write("**Trạng thái:** Nhân viên đang có tâm lý ổn định.")
            
            # Debug: Hiển thị giá trị input đã scale để kiểm tra nếu cần
            with st.expander("Xem chi tiết dữ liệu nạp vào Model"):
                st.dataframe(df_input)

    except Exception as e:
        st.error(f"Lỗi khi dự báo: {e}")