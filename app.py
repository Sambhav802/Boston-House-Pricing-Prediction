import pickle
import streamlit as st
import numpy as np

# Load model and scaler
model = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🏠 Boston House Price Prediction App")

# Input fields
st.subheader("📝 Please Enter Property Details:")
CRIM = st.number_input("🚨 Enter CRIM value (Crime Rate):", min_value=0.0, step=0.01)
ZN = st.number_input("🏘️ Enter ZN value (Residential Zone):", min_value=0.0, step=0.1)
INDUS = st.number_input("🏭 Enter INDUS value (Industrial):", min_value=0.0, step=0.1)
CHAS = st.selectbox("🌊 CHAS (Charles River dummy variable)", options=[0, 1])
NOX = st.number_input("💨 Enter NOX value (Air Pollution):", min_value=0.0, max_value=1.0, step=0.01)
RM = st.number_input("🛏️ Enter RM value (Average Rooms):", min_value=0.0, step=0.1)
AGE = st.number_input("⏳ Enter AGE value:", min_value=0.0, step=0.1)
DIS = st.number_input("🚶 Enter DIS value (Distance):", min_value=0.0, step=0.1)
RAD = st.number_input("🛣️ Enter RAD value (Highway Access):", min_value=1.0, step=1.0)
TAX = st.number_input("💰 Enter TAX value:", min_value=0.0, step=1.0)
PTRATIO = st.number_input("📚 Enter PTRATIO value (Student-Teacher Ratio):", min_value=0.0, step=0.1)
B = st.number_input("📊 Enter B value:", min_value=0.0, step=0.1)
LSTAT = st.number_input("📉 Enter LSTAT value:", min_value=0.0, step=0.1)

if st.button("🎯 Predict"):
    # ...existing code...
    input_data = np.array([[CRIM, ZN, INDUS, NOX, RM, AGE, DIS,
                            RAD, TAX, PTRATIO, B, LSTAT]])
    
    scaled_data = scaler.transform(input_data)
    final_input = np.insert(scaled_data, 3, CHAS, axis=1)
    prediction = model.predict(final_input)[0]

    st.success(f"💵 Predicted House Price: ${prediction * 1000:,.2f}")

# Footer
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit | By Sambhav Sharma (👨‍💻 AI/ML Enthusiast)")