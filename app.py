import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 页面标题
st.title("心脏病预测系统（随机森林模型）")

# 用户输入界面
st.sidebar.header("输入病人信息")

def user_input_features():
    age = st.sidebar.slider("年龄", 20, 90, 50)
    sex = st.sidebar.selectbox("性别", ["male", "female"])
    cp = st.sidebar.selectbox("胸痛类型", ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
    trestbps = st.sidebar.slider("静息血压", 80, 200, 130)
    chol = st.sidebar.slider("胆固醇", 100, 400, 250)
    fbs = st.sidebar.selectbox("空腹血糖 > 120mg/dl", ["False", "True"])
    restecg = st.sidebar.selectbox("静息心电图结果", ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"])
    thalach = st.sidebar.slider("最大心率", 60, 220, 150)
    exang = st.sidebar.selectbox("运动诱发心绞痛", ["No", "Yes"])
    oldpeak = st.sidebar.slider("ST 抑制值", 0.0, 6.0, 1.0, step=0.1)
    slope = st.sidebar.selectbox("ST 斜率类型", ["upsloping", "flat", "downsloping"])
    ca = st.sidebar.slider("主要血管数量", 0, 3, 0)
    thal = st.sidebar.selectbox("地中海贫血类型", ["normal", "fixed defect", "reversable defect"])

    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 加载原始数据，准备编码器
df = pd.read_csv("heart_disease_uci.csv")

# ✅ 删除这一行，因为 CSV 没有这些列
# df = df.drop(columns=["id", "dataset"])

label_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
le_dict = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# 应用相同编码器到输入数据
for col in label_cols:
    input_df[col] = le_dict[col].transform(input_df[col])

# 显示输入
st.subheader("病人信息预览")
st.write(input_df)

# 加载模型
model = joblib.load("model.pkl")

# 预测结果
prediction = model.predict(input_df)[0]
st.subheader("预测结果")
st.write("有心脏病" if prediction == 1 else "没有心脏病")
