from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# 현재 작업 디렉토리 기준
current_dir = Path.cwd()

model = joblib.load(current_dir / "model.pkl")
le_gender = joblib.load(current_dir / "le_gender.pkl")
le_product = joblib.load(current_dir / "le_product.pkl")
le_channel = joblib.load(current_dir / "le_channel.pkl")
le_reason = joblib.load(current_dir / "le_reason.pkl")

st.title("구독 이탈 예측기")
st.write("고객 정보를 입력하면 이탈 확률을 예측합니다.")

# 사용자 입력
age = st.number_input("고객 현재 나이", min_value=10, max_value=100, value=40)
gender = st.selectbox("성별", ["male", "female"])
product = st.selectbox("상품", ["prd_1", "prd_2"])
signup_year = st.selectbox("가입 연도", list(range(2017, 2025)))
signup_month = st.selectbox("가입 월", list(range(1, 13)))
subscription_duration = st.number_input("사용 일수 (subscription_duration)", min_value=0, max_value=2000, value=30)
total_inquiries = st.number_input("총 문의 수", min_value=0, value=1)
channel = st.selectbox("문의 채널", ["phone", "email", "No Inquiry"])
reason = st.selectbox("문의 사유", ["signup", "support", "No Inquiry"])

# 파생 변수
signup_age = age - (datetime.now().year - signup_year)
signup_age = max(signup_age, 0)

# 입력값을 데이터프레임으로 정리
input_data = pd.DataFrame([{
    "age": age,
    "gender": le_gender.transform([gender])[0],
    "product": le_product.transform([product])[0],
    "signup_year": signup_year,
    "signup_month": signup_month,
    "subscription_duration": subscription_duration,
    "total_inquiries": total_inquiries,
    "channel": le_channel.transform([channel])[0],
    "reason": le_reason.transform([reason])[0],
    "signup_age": signup_age
}])

# 예측
if st.button("이탈률 예측하기"):
    churn_proba = model.predict_proba(input_data)[:, 1][0]
    churn_class = model.predict(input_data)[0]
    
    st.subheader("📊 예측 결과")
    st.write(f"**이탈 확률:** {churn_proba:.2%}")
    st.write(f"**예측 결과:** {'이탈 가능성 높음 🚨' if churn_class == 1 else '이탈 가능성 낮음 ✅'}")
