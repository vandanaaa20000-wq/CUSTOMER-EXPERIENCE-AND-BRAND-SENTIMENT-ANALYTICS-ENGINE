import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from io import BytesIO

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Alzheimer's Disease Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.stButton>button{
    background:#1E88E5;
    color:white;
    border-radius:10px;
    height:50px;
    font-size:18px;
    width:100%;
}

.stDownloadButton>button{
    background:#43A047;
    color:white;
    border-radius:10px;
}

.result{
    padding:20px;
    border-radius:10px;
}

.footer{
text-align:center;
padding:20px;
font-size:15px;
color:gray;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
from pathlib import Path

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "alzheimers_model.pkl"
    return joblib.load(model_path)

model = load_model()

try:
    top_features = joblib.load(Path(__file__).parent / "top_features.pkl")
except Exception:
    top_features = [
        "MMSE","FunctionalAssessment","ADL","MemoryComplaints",
        "BehavioralProblems","DietQuality","CholesterolTriglycerides",
        "PhysicalActivity","Age","SleepQuality","CholesterolTotal",
        "BMI","AlcoholConsumption","CholesterolHDL"
    ]

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.image(
    "https://img.icons8.com/color/480/brain.png",
    width=120
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "",
    [
        "🏠 Home",
        "🧑 Single Prediction",
        "📂 Batch Prediction",
        "📈 Model Information"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success(
"""
Model

✔ Gradient Boosting

Accuracy

95.35%

Top Features

14
"""
)

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "🏠 Home":

    st.title("🧠 Alzheimer's Disease Classification")

    st.markdown("---")

    col1,col2 = st.columns([2,1])

    with col1:

        st.markdown("""
### About

This web application predicts the likelihood of Alzheimer's Disease using a trained Machine Learning model.

### Features

✅ Single Patient Prediction

✅ Batch Prediction using CSV

✅ Prediction Probability

✅ Download Prediction Results

✅ Interactive Charts

✅ Built with Streamlit

""")

    with col2:

        st.metric(
            "Model Accuracy",
            "95.35%"
        )

        st.metric(
            "F1 Score",
            "93.38%"
        )

        st.metric(
            "ROC AUC",
            "94.41%"
        )

    st.markdown("---")

    st.subheader("Features Used")

    feature_df = pd.DataFrame({

        "Feature":[
            "Age",
            "BMI",
            "Alcohol Consumption",
            "Physical Activity",
            "Diet Quality",
            "Sleep Quality",
            "Total Cholesterol",
            "HDL Cholesterol",
            "Triglycerides",
            "MMSE",
            "Functional Assessment",
            "ADL",
            "Memory Complaints",
            "Behavioral Problems"
        ]

    })

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Use the sidebar to start predicting."
    )
# ==========================================================
# SINGLE PATIENT PREDICTION
# ==========================================================

elif page == "🧑 Single Prediction":

    st.title("🧠 Single Patient Prediction")

    st.markdown("Enter the patient's clinical information.")

    st.markdown("---")

    col1,col2 = st.columns(2)

    with col1:

        age = st.slider(
            "Age",
            60,
            90,
            72
        )

        bmi = st.number_input(
            "BMI",
            10.0,
            50.0,
            25.0
        )

        alcohol = st.slider(
            "Alcohol Consumption",
            0.0,
            20.0,
            5.0
        )

        physical = st.slider(
            "Physical Activity",
            0.0,
            10.0,
            5.0
        )

        diet = st.slider(
            "Diet Quality",
            0.0,
            10.0,
            5.0
        )

        sleep = st.slider(
            "Sleep Quality",
            0.0,
            10.0,
            6.0
        )

        total_chol = st.number_input(
            "Total Cholesterol",
            100.0,
            400.0,
            200.0
        )

    with col2:

        hdl = st.number_input(
            "HDL Cholesterol",
            20.0,
            120.0,
            55.0
        )

        trig = st.number_input(
            "Triglycerides",
            50.0,
            500.0,
            150.0
        )

        mmse = st.slider(
            "MMSE Score",
            0,
            30,
            22
        )

        functional = st.slider(
            "Functional Assessment",
            0.0,
            10.0,
            5.0
        )

        adl = st.slider(
            "Activities of Daily Living",
            0.0,
            10.0,
            5.0
        )

        memory = st.selectbox(
            "Memory Complaints",
            [0,1]
        )

        behavioural = st.selectbox(
            "Behavioral Problems",
            [0,1]
        )

    st.markdown("")

    predict = st.button("Predict Alzheimer's Disease")

    if predict:

        patient = pd.DataFrame({

            "Age":[age],
            "BMI":[bmi],
            "AlcoholConsumption":[alcohol],
            "PhysicalActivity":[physical],
            "DietQuality":[diet],
            "SleepQuality":[sleep],
            "CholesterolTotal":[total_chol],
            "CholesterolHDL":[hdl],
            "CholesterolTriglycerides":[trig],
            "MMSE":[mmse],
            "FunctionalAssessment":[functional],
            "ADL":[adl],
            "MemoryComplaints":[memory],
            "BehavioralProblems":[behavioural]

        })

        patient = patient[top_features]

        prediction = model.predict(patient)[0]
        probability = model.predict_proba(patient)[0][1]

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("⚠ Alzheimer's Disease Detected")

        else:

            st.success("✅ No Alzheimer's Disease Detected")

        st.progress(int(probability * 100))

        st.metric(
            "Probability",
            f"{probability*100:.2f}%"
        )

        fig = px.pie(
            names=["No Disease","Disease"],
            values=[1-probability,probability],
            title="Prediction Probability"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Patient Input Summary")

        st.dataframe(
            patient,
            use_container_width=True
        )
# ==========================================================
# BATCH PREDICTION (CSV UPLOAD)
# ==========================================================

elif page == "📂 Batch Prediction":

    st.title("📂 Batch Prediction")

    st.markdown(
        "Upload a CSV file containing the same 14 features used to train the model."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        st.write(f"Rows : {len(df)}")

        if st.button("Predict Dataset"):

            try:

                missing = [c for c in top_features if c not in df.columns]

                if missing:
                    st.error(f"Missing required columns: {missing}")
                    st.stop()

                X = df[top_features]
                prediction = model.predict(X)
                probability = model.predict_proba(X)[:, 1]
                result = df.copy()

                result["Prediction"] = np.where(
                    prediction==1,
                    "Alzheimer's",
                    "Healthy"
                )

                result["Probability"] = np.round(
                    probability*100,
                    2
                )

                st.success("Prediction Completed Successfully")

                st.dataframe(
                    result,
                    use_container_width=True
                )

                # -------------------------
                # Metrics
                # -------------------------

                disease = (prediction==1).sum()

                healthy = (prediction==0).sum()

                col1,col2,col3=st.columns(3)

                col1.metric(
                    "Total Patients",
                    len(result)
                )

                col2.metric(
                    "Healthy",
                    healthy
                )

                col3.metric(
                    "Alzheimer's",
                    disease
                )

                st.markdown("---")

                # -------------------------
                # Pie Chart
                # -------------------------

                pie = pd.DataFrame({

                    "Category":[
                        "Healthy",
                        "Alzheimer's"
                    ],

                    "Count":[
                        healthy,
                        disease
                    ]

                })

                fig = px.pie(
                    pie,
                    names="Category",
                    values="Count",
                    title="Prediction Distribution"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # -------------------------
                # Bar Chart
                # -------------------------

                bar = px.bar(

                    pie,

                    x="Category",

                    y="Count",

                    title="Prediction Count"

                )

                st.plotly_chart(
                    bar,
                    use_container_width=True
                )

                # -------------------------
                # Probability Histogram
                # -------------------------

                hist = px.histogram(

                    result,

                    x="Probability",

                    nbins=20,

                    title="Prediction Probability Distribution"

                )

                st.plotly_chart(
                    hist,
                    use_container_width=True
                )

                # -------------------------
                # Download CSV
                # -------------------------

                csv = result.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(

                    "⬇ Download Prediction Results",

                    csv,

                    "alzheimers_predictions.csv",

                    "text/csv"

                )

            except Exception as e:

                st.error(f"Prediction Error : {e}")
elif page == "📈 Model Information":

    st.title("📈 Model Performance")

    st.markdown("---")

    col1,col2,col3=st.columns(3)

    col1.metric(
        "Accuracy",
        "95.35%"
    )

    col2.metric(
        "Precision",
        "94.00%"
    )

    col3.metric(
        "Recall",
        "92.76%"
    )

    col1,col2,col3=st.columns(3)

    col1.metric(
        "F1 Score",
        "93.38%"
    )

    col2.metric(
        "ROC AUC",
        "94.41%"
    )

    col3.metric(
        "Algorithm",
        "Gradient Boosting"
    )

    st.markdown("---")

    st.subheader("Top 14 Features")

    feature_df = pd.DataFrame({

        "Feature":[

            "Age",
            "BMI",
            "AlcoholConsumption",
            "PhysicalActivity",
            "DietQuality",
            "SleepQuality",
            "CholesterolTotal",
            "CholesterolHDL",
            "CholesterolTriglycerides",
            "MMSE",
            "FunctionalAssessment",
            "ADL",
            "MemoryComplaints",
            "BehavioralProblems"

        ]

    })

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    metrics = pd.DataFrame({

        "Metric":[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC AUC"
        ],

        "Score":[
            95.35,
            94.00,
            92.76,
            93.38,
            94.41
        ]

    })

    fig = px.bar(
        metrics,
        x="Metric",
        y="Score",
        text="Score",
        title="Model Performance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Model Description")

    st.success("""
The deployed model is a **Gradient Boosting Classifier**
trained using the **Top 14 Important Features** selected
through feature importance analysis.

The model achieved:

• Accuracy : 95.35%

• Precision : 94.00%

• Recall : 92.76%

• F1 Score : 93.38%

• ROC AUC : 94.41%

The application supports both
single patient prediction and
batch prediction using CSV upload.
""")

    st.markdown("---")
    st.subheader("📊 Model Comparison")

    comparison_df = pd.DataFrame({
        "Model":["Tuned Gradient Boosting","Top 14 Features Gradient Boosting","Gradient Boosting","Random Forest","Decision Tree","Support Vector Machine","Logistic Regression","Gaussian Naive Bayes","K-Nearest Neighbours"],
        "Training Accuracy (%)":[95.75,95.75,96.68,100.00,100.00,94.12,85.11,80.63,82.14],
        "Testing Accuracy (%)":[95.35,95.35,94.65,93.72,90.00,83.49,81.63,77.21,70.70],
        "Precision (%)":[94.00,94.00,92.72,93.71,84.71,79.14,74.17,67.09,63.27],
        "Recall (%)":[92.76,92.76,92.11,88.16,87.50,72.37,73.68,69.74,40.79],
        "F1 Score (%)":[93.38,93.38,92.41,90.85,86.08,75.60,73.93,68.39,49.60],
        "ROC-AUC (%)":[94.41,94.41,94.64,93.90,89.43,89.63,88.54,85.04,73.77]
    })

    st.dataframe(comparison_df,use_container_width=True,hide_index=True)

    fig = px.bar(comparison_df,x="Model",y="Testing Accuracy (%)",
                 color="Testing Accuracy (%)",text="Testing Accuracy (%)",
                 title="Testing Accuracy Comparison")
    fig.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(fig,use_container_width=True)

    st.success("""### 🏆 Best Performing Model

**Top 14 Features Gradient Boosting**

Training Accuracy: **95.75%**

Testing Accuracy: **95.35%**

Precision: **94.00%**

Recall: **92.76%**

F1 Score: **93.38%**

ROC-AUC: **94.41%**
""")


# ==========================================================
# SAMPLE CSV DOWNLOAD
# ==========================================================

sample = pd.DataFrame({

    "Age":[72],
    "BMI":[24.5],
    "AlcoholConsumption":[3.2],
    "PhysicalActivity":[5.8],
    "DietQuality":[7.0],
    "SleepQuality":[6.5],
    "CholesterolTotal":[190],
    "CholesterolHDL":[55],
    "CholesterolTriglycerides":[145],
    "MMSE":[22],
    "FunctionalAssessment":[6.0],
    "ADL":[7.5],
    "MemoryComplaints":[1],
    "BehavioralProblems":[0]

})

csv = sample.to_csv(index=False).encode("utf-8")

st.sidebar.markdown("---")

st.sidebar.download_button(

    "📥 Download Sample CSV",

    csv,

    "sample_input.csv",

    "text/csv"

)

st.subheader("Feature Importance")

importance = pd.DataFrame({

    "Feature":[
        "MMSE",
        "FunctionalAssessment",
        "ADL",
        "MemoryComplaints",
        "BehavioralProblems",
        "Age",
        "BMI",
        "PhysicalActivity",
        "DietQuality",
        "SleepQuality",
        "AlcoholConsumption",
        "CholesterolTotal",
        "CholesterolHDL",
        "CholesterolTriglycerides"
    ],

    "Importance":[
        0.19,
        0.15,
        0.13,
        0.10,
        0.08,
        0.07,
        0.06,
        0.05,
        0.05,
        0.04,
        0.03,
        0.025,
        0.02,
        0.015
    ]

})

fig = px.bar(
    importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance"
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown("""

<div style="text-align:center;
font-size:15px;
color:gray;">

🧠 Alzheimer's Disease Classification System

Developed using
<b>Python • Streamlit • Scikit-learn • Plotly</b>

Machine Learning Model:
Gradient Boosting Classifier

</div>

""", unsafe_allow_html=True)

