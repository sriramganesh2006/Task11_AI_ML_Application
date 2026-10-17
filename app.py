from pathlib import Path
import pickle

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "student_model.pkl"

FEATURES = [
    "Marks",
    "StudyHours",
    "Attendance",
    "AssignmentsCompleted",
]

st.set_page_config(
    page_title="AI Student Performance Analyzer",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


st.title("🎓 AI Student Performance Analyzer")
st.write(
    "An educational machine-learning application "
    "for exploring student performance."
)

st.warning(
    "This demonstration uses a small sample dataset. "
    "Predictions are not guaranteed academic outcomes."
)

st.sidebar.header("Student Details")

marks = st.sidebar.slider("Marks", 0, 100, 75)
hours = st.sidebar.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80)
assignments = st.sidebar.slider("Assignments completed", 0, 10, 7)

if st.sidebar.button("Analyze Performance", type="primary"):
    try:
        model = load_model()

        student = pd.DataFrame(
            [[marks, hours, attendance, assignments]],
            columns=FEATURES,
        )

        prediction = model.predict(student)[0]

        st.subheader("Analysis Result")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Marks", f"{marks}/100")
        col2.metric("Study Hours", f"{hours} hrs/day")
        col3.metric("Attendance", f"{attendance}%")
        col4.metric("Assignments", f"{assignments}/10")

        if prediction == "Pass":
            st.success("Model prediction: Pass")
        else:
            st.info("Model prediction: Fail")

        st.subheader("Performance Visualization")

        chart_data = pd.DataFrame({
            "Metric": ["Marks", "Attendance", "Assignments"],
            "Value": [marks, attendance, assignments * 10],
        })

        fig, ax = plt.subplots()
        ax.bar(chart_data["Metric"], chart_data["Value"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Score (%)")
        ax.set_title("Student Performance Inputs")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Study Insights")

        if hours < 2:
            st.write("Try building a consistent daily study routine.")
        else:
            st.write("Continue revising regularly and practising questions.")

        if attendance < 75:
            st.write("Improving attendance may help maintain learning continuity.")

        if assignments < 6:
            st.write("Try completing practice assignments consistently.")

    except FileNotFoundError:
        st.error(
            "Trained model not found. Run `py train_model.py` first."
        )
    except Exception as error:
        st.error(f"Could not generate prediction: {error}")

else:
    st.info(
        "Choose the student details in the sidebar, "
        "then click Analyze Performance."
    )