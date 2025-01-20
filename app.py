import streamlit as st
import pandas as pd
import pickle

# Set page configuration
st.set_page_config(page_title="Campus Placement Predictor", page_icon="🎓", layout="centered")

# Custom CSS for background color and styling
st.markdown(r"""
    <style>
    body {
        background-color: maroon;
        color: white;
    }
    .title {
        text-align: center;
        color: white;
        font-size: 2.5em;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Add university image above the title
st.image("university-school-vector-flat-illustration-260nw-2304229867.jpg", width=600)  # Replace with your image filename
st.markdown("<h1 class='title'>Campus Placement Predictor</h1>", unsafe_allow_html=True)

st.write("""
This app predicts whether a student will be placed based on their details.
Fill in the form below to see the prediction.
""")

# User inputs
age = st.number_input("Age", min_value=17, max_value=30, value=21)
gender = st.selectbox("Gender", options=["Male", "Female"])
stream = st.selectbox("Stream", options=[
    "Electronics And Communication", 
    "Computer Science", 
    "Information Technology", 
    "Mechanical", 
    "Electrical", 
    "Civil"
])
internships = st.slider("Number of Internships", 0, 5, 1)
cgpa = st.number_input("CGPA", min_value=5.0, max_value=10.0, value=7.5, step=0.1)
hostel = st.radio("Lives in Hostel?", ["Yes", "No"])
backlogs = st.slider("Number of Backlogs", 0, 10, 0)

# Preprocess inputs
gender = 1 if gender == "Male" else 0
stream_mapping = {
    "Electronics And Communication": 1,
    "Computer Science": 2,
    "Information Technology": 3,
    "Mechanical": 4,
    "Electrical": 5,
    "Civil": 6
}
stream = stream_mapping[stream]
hostel = 1 if hostel == "Yes" else 0

# Make prediction
if st.button("Predict Placement"):
    features = [[age, gender, stream, internships, cgpa, hostel, backlogs]]
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.success("🎉 Congratulations! The student is likely to get placed.")
    else:
        st.error("❌ The student is unlikely to get placed.")

# Footer
st.write("This predictor is powered by a trained machine learning model.")
