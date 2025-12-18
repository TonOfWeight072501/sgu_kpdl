import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. CẤU HÌNH TRANG & LOAD MODEL
st.set_page_config(
    page_title="Psychological HR Analytics",
    layout="wide"
)

@st.cache_resource
def load_model():
    # Load model tốt nhất
    try:
        model = joblib.load('best_ensemble_model.pkl')
        return model
    except FileNotFoundError:
        return None

model = load_model()

# 2. CẤU HÌNH SCALING

# REAL_CONFIG: Dùng để chuyển đổi số thực (VD: Tuổi 30) sang thang đo 0-1
REAL_CONFIG = {
    'Age': {'min': 18, 'max': 60},
    'Years_Experience': {'min': 0, 'max': 40},
    'MonthlySalary': {'min': 2000, 'max': 30000}, 
    'Allowances': {'min': 0, 'max': 5000},
    'Distance_to_work': {'min': 0, 'max': 50}, 
    'Training_programs': {'min': 0, 'max': 10}
}

def normalize(value, col_name):
    """Hàm chuyển đổi số thực sang chuẩn 0-1 cho Model"""
    if col_name in REAL_CONFIG:
        min_val = REAL_CONFIG[col_name]['min']
        max_val = REAL_CONFIG[col_name]['max']
        # Công thức Min-Max Scaling
        norm_val = (value - min_val) / (max_val - min_val)
        # Đảm bảo không vượt quá 0-1 (clip)
        return np.clip(norm_val, 0.0, 1.0)
    return value

# 3. GIAO DIỆN NHẬP LIỆU (NHẬP SỐ THỰC)

st.title("DỰ BÁO NGHỈ VIỆC")
st.markdown("---")

if model is None:
    st.error("Lỗi: Không tìm thấy file 'best_ensemble_model.pkl'.")
    st.stop()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Thông tin Hồ sơ Nhân viên")
    
    # NHÓM 1: THÔNG TIN CƠ BẢN (NHẬP SỐ THỰC)
    with st.expander("1. Thông tin cơ bản & Thu nhập", expanded=True):
        
        # 1. Tuổi (Nhập số tuổi thật)
        age_input = st.number_input("Độ tuổi", 
                                    min_value=18, max_value=65, value=30, step=1)
        
        # 2. Kinh nghiệm (Nhập số năm)
        c_exp, c_sal = st.columns(2)
        with c_exp:
            exp_input = st.number_input("Kinh nghiệm (Năm)", 0, 40, 5, step=1)
            allowance_input = st.number_input("Phụ cấp (Số tiền)", 0, 10000, 500, step=100)
        
        # 3. Lương (Nhập số tiền)
        with c_sal:
            salary_input = st.number_input("Mức lương (Số tiền)", 0, 50000, 5000, step=500)
            training_input = st.number_input("Số khóa đào tạo (3 năm)", 0, 20, 2, step=1)

        # 4. Bằng cấp (Map từ chữ sang số)
        degree_map = {"Bachelor (Cử nhân)": 0.33, "Master (Thạc sĩ)": 0.66, "PhD (Tiến sĩ)": 1.0, "Không có": 0.0}
        degree_choice = st.selectbox("Bằng cấp", list(degree_map.keys()))
        academic_val = degree_map[degree_choice]

    # NHÓM 2: DỮ LIỆU TÂM LÝ
    # Tâm lý là định tính, dùng thanh kéo 0-1 hoặc thang điểm 1-10 là hợp lý nhất.
    with st.expander("2. Sức khỏe Tâm lý & Cam kết", expanded=True):
        st.info("Thang điểm: 0 (Thấp) - 10 (Cao)")
        
        # Chuyển thanh kéo thành thang 10 cho dễ hiểu, sau đó chia 10 để về 0-1
        psy_exhaustion = st.slider("Kiệt quệ tâm lý (Burnout)", 0, 10, 5) / 10.0
        phys_stress = st.slider("Căng thẳng thể chất", 0, 10, 4) / 10.0
        emo_commit = st.slider("Cam kết cảm xúc", 0, 10, 6) / 10.0
        job_eng = st.slider("Sự gắn kết công việc", 0, 10, 6) / 10.0
        
    # NHÓM 3: MÔI TRƯỜNG
    with st.expander("3. Môi trường & Sự ổn định"):
        job_stability = st.slider("Sự ổn định công việc", 0, 10, 7) / 10.0
        
        c1, c2 = st.columns(2)
        with c1:
            env_sat = st.slider("Hài lòng môi trường", 0, 10, 6) / 10.0
            job_sat = st.slider("Hài lòng công việc", 0, 10, 6) / 10.0
        with c2:
            wlb = st.slider("Cân bằng cuộc sống", 0, 10, 5) / 10.0
            job_opp = st.slider("Cơ hội nghề nghiệp", 0, 10, 5) / 10.0
        
    # NHÓM 4: KHÁC
    with st.expander("4. Các yếu tố khác"):
        # Khoảng cách nhập Km thật
        dist_input = st.number_input("Khoảng cách đi làm (Km)", 0.0, 100.0, 10.0, step=0.5)
        
        job_support = st.slider("Hỗ trợ công việc (0-10)", 0, 10, 6) / 10.0
        promotion = st.selectbox("Được thăng chức xứng đáng", ["Không", "Có"])
        prom_val = 1.0 if promotion == "Có" else 0.0
        
        st.caption("Tham số kỹ thuật:")
        job_freq = st.number_input("Tần suất chức danh", 0.0, 1.0, 0.2)
        stab_ratio = st.number_input("Tỷ lệ ổn định", 0.0, 1.0, 0.5)

    # CHUYỂN ĐỔI SỐ THỰC VỀ 0-1
    input_data = {
        'Age': normalize(age_input, 'Age'),
        'Academic_degree': academic_val,
        'Years_Experience': normalize(exp_input, 'Years_Experience'),
        'MonthlySalary': normalize(salary_input, 'MonthlySalary'),
        'Allowances': normalize(allowance_input, 'Allowances'),
        'Get_ Deserved_Promotion': prom_val, 
        'Training_programs_ During_last_three_years': normalize(training_input, 'Training_programs'), 
        'Job_Support': job_support,
        'Emotional_Commitment': emo_commit,
        'Job_Engagement': job_eng,
        'Distance_to_work': normalize(dist_input, 'Distance_to_work'),
        'Work_Live_Balance': wlb,
        'Physical_Stress': phys_stress,
        'Psychological_Exhaustion': psy_exhaustion,
        'Job_Stability': job_stability,
        'Environment_Satisfaction': env_sat,
        'Job_Satisfaction': job_sat,
        'Job_Opportunities': job_opp,
        'JobTitle_Freq': job_freq,
        'Stability_Ratio': stab_ratio
    }

# 4. HIỂN THỊ KẾT QUẢ

with col2:
    st.subheader("Kết quả Phân tích")
    
    # Tạo DataFrame từ dữ liệu đã chuẩn hóa
    df_input = pd.DataFrame([input_data])
    
    with st.expander("Xem dữ liệu đầu vào (Đã quy đỗi về 0-1)"):
        st.dataframe(df_input.T)

    if st.button("Thực hiện Dự báo", type="primary"):
        with st.spinner('Đang xử lý dữ liệu...'):
            try:
                # Dự báo
                pred_proba = model.predict_proba(df_input)[0]
                prob_leave = pred_proba[1] * 100
                
                st.markdown(f"### Xác suất Nghỉ việc: **{prob_leave:.2f}%**")
                st.progress(int(prob_leave))
                
                if prob_leave > 50:
                    st.error("CẢNH BÁO: MỨC ĐỘ RỦI RO CAO")
                    st.write("---")
                    st.write("**Các yếu tố rủi ro chính:**")
                    
                    if psy_exhaustion > 0.6:
                        st.write("- Kiệt quệ tâm lý (Burnout): Mức độ cao.")
                    if emo_commit < 0.4:
                        st.write("- Cam kết cảm xúc: Thấp.")
                    if job_stability < 0.4:
                        st.write("- Sự ổn định công việc: Thấp.")
                    if salary_input < 5000: # Logic kết hợp số thực
                         st.write(f"- Lương thấp ({salary_input}): Cần xem xét.")

                    st.warning("Kiến nghị: Cần xem xét các biện pháp can thiệp.")
                    
                else:
                    st.success("TRẠNG THÁI: ỔN ĐỊNH")
                    st.write("Nhân viên có chỉ số tâm lý tốt.")
            
            except Exception as e:
                st.error(f"Lỗi xử lý: {e}")
                st.info("Vui lòng kiểm tra lại dữ liệu đầu vào.")