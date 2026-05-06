import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

with open('student_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Student Performance Prediction System")

tab1, tab2, tab3 = st.tabs(["Individual Prediction", "Performance Metrics", "Trend Analysis"])

with tab1:
    st.header("Predict Student Risk")
    study = st.number_input("Daily Study Hours", value=5.0)
    attendance = st.number_input("Attendance %", value=85.0)
    prev_marks = st.number_input("Previous Marks", value=70.0)
    assignments = st.number_input("Assignments", value=80.0)
    internals = st.number_input("Internal Marks", value=75.0)

    if st.button("Predict"):
        input_data = np.array([[study, attendance, prev_marks, assignments, internals]])
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("Result: PASS")
        else:
            st.error("Result: AT RISK")

with tab2:
    st.header("Model Performance Evaluation")
    # These are the fixed metrics from your notebook evaluation
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "92%")
    col2.metric("Precision", "0.91")
    col3.metric("F1-Score", "0.92")
    
    st.write("The model was trained using 100,000 student records and evaluated using a 20% test split.")

with tab3:
    st.header("Dataset Trends")
    df = pd.read_csv('student_performance_dataset_req.csv')
    
    st.subheader("Attendance vs Final Result")
    fig, ax = plt.subplots()
    sns.boxplot(x='Final_Result', y='Attendance', data=df, ax=ax)
    st.pyplot(fig)