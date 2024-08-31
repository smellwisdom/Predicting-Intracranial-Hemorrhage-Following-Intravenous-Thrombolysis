import streamlit as st
import pandas as pd
import joblib

# 自定义CSS来扩大内容宽度
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 90%;
        padding-left: 5%;
        padding-right: 5%;
    }
    .force-plot {
        border: none;
        width: 100%;
        overflow-x: scroll;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 加载训练好的模型
model_path = 'final_model.pkl'  # 更新模型路径为实际保存的模型文件名
loaded_model = joblib.load(model_path)

# 原始特征名称（与模型训练时使用的一致）
original_feature_names = [
    'Gender', 'Age', 'PostAwakening_Stroke', 'InHospital_Stroke', 'BMI',
    'Systolic_BP', 'Diastolic_BP', 'Admission_mRS_Score', 'Admission_NIHSS_Score',
    'Swallowing_Function_Score', 'Onset_To_Needle_Time', 'Antiplatelet_Therapy',
    'Anticoagulation_Therapy', 'TOAST_Classification_1', 'TOAST_Classification_3',
    'TOAST_Classification_2', 'TOAST_Classification_5', 'TOAST_Classification_4'
]

# 映射关系更新，将用户友好的名称映射回模型的特征名称
feature_name_mapping = {
    'Gender': 'Gender',
    'Age': 'Age',
    'Post_Awakening_Stroke': 'PostAwakening_Stroke',
    'In_Hospital_Stroke': 'InHospital_Stroke',
    'Body_Mass_Index': 'BMI',
    'Systolic_Blood_Pressure': 'Systolic_BP',
    'Diastolic_Blood_Pressure': 'Diastolic_BP',
    'Admission_mRS_Score': 'Admission_mRS_Score',
    'Admission_NIHSS_Score': 'Admission_NIHSS_Score',
    'Swallowing_Function_Score': 'Swallowing_Function_Score',
    'Onset_To_Needle_Time': 'Onset_To_Needle_Time',
    'Antiplatelet_Therapy': 'Antiplatelet_Therapy',
    'Anticoagulation_Therapy': 'Anticoagulation_Therapy',
    'TOAST_LAA': 'TOAST_Classification_1',
    'TOAST_SAA': 'TOAST_Classification_3',
    'TOAST_CE': 'TOAST_Classification_2',
    'TOAST_SUE': 'TOAST_Classification_5',
    'TOAST_SOE': 'TOAST_Classification_4'
}

# Streamlit 应用程序接口
# st.title("Stroke Outcome Prediction")

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.centered {
    display: flex;
    justify-content: center;
    align-items: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Input Patient Details:</p>', unsafe_allow_html=True)

# 创建输入表单布局
col1, col2 = st.columns(2)

with col1:
    input_data = {}
    input_data['Gender'] = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
    input_data['Age'] = st.slider('Age', min_value=0, max_value=100, value=60, step=1)
    input_data['Post_Awakening_Stroke'] = st.selectbox('Post Awakening Stroke', options=[0, 1])
    input_data['In_Hospital_Stroke'] = st.selectbox('In Hospital Stroke', options=[0, 1])
    input_data['Body_Mass_Index'] = st.slider('Body Mass Index', min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    input_data['Systolic_Blood_Pressure'] = st.slider('Systolic Blood Pressure', min_value=80, max_value=240, value=120, step=1)
    input_data['Diastolic_Blood_Pressure'] = st.slider('Diastolic Blood Pressure', min_value=40, max_value=140, value=80, step=1)
    input_data['Admission_mRS_Score'] = st.slider('Admission mRS Score', min_value=0, max_value=6, value=2, step=1)
    input_data['Admission_NIHSS_Score'] = st.slider('Admission NIHSS Score', min_value=0, max_value=40, value=5, step=1)

with col2:
    input_data['Swallowing_Function_Score'] = st.slider('Swallowing Function Score', min_value=0, max_value=5, value=3,step=1)
    input_data['Onset_To_Needle_Time'] = st.slider('Onset To Needle Time', min_value=0, max_value=7200, value=120,step=1)
    input_data['Antiplatelet_Therapy'] = st.selectbox('Antiplatelet Therapy', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    input_data['Anticoagulation_Therapy'] = st.selectbox('Anticoagulation Therapy', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    input_data['TOAST_LAA'] = st.selectbox('TOAST Classification: LAA', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    input_data['TOAST_SAA'] = st.selectbox('TOAST Classification: SAA', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    input_data['TOAST_CE'] = st.selectbox('TOAST Classification: CE', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    input_data['TOAST_SUE'] = st.selectbox('TOAST Classification: SUE', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    input_data['TOAST_SOE'] = st.selectbox('TOAST Classification: SOE', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# 将输入数据转换为 DataFrame
input_df = pd.DataFrame([input_data])

# 映射用户友好的名称回模型训练时的名称
input_df.rename(columns=feature_name_mapping, inplace=True)

# 确保列顺序与模型训练时相同
input_df = input_df[original_feature_names]

# 进行预测
if st.button('Predict'):
    prediction_prob = loaded_model.predict_proba(input_df)[0, 1]
    prediction_text = f"Based on feature values, predicted probability of ICH is {prediction_prob * 100:.2f}%"
    st.markdown(f'<p style="font-size:45px; font-weight: bold;">{prediction_text}</p>', unsafe_allow_html=True)

