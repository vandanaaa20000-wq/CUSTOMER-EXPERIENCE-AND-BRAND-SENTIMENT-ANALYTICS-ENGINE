# ==========================================================
# CUSTOMER REVIEW SENTIMENT ANALYSIS - STREAMLIT APPLICATION
# ==========================================================

import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import plotly.express as px

from pathlib import Path
from io import BytesIO

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Customer Review Sentiment Analysis",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7fa;
    }

    .stButton > button {
        background: #1E88E5;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-size: 17px;
        font-weight: 600;
        width: 100%;
        border: none;
    }

    .stDownloadButton > button {
        background: #43A047;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 600;
        border: none;
    }

    .result-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-top: 10px;
        font-size: 20px;
        font-weight: 600;
    }

    .footer {
        text-align: center;
        padding: 20px;
        font-size: 14px;
        color: gray;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# DOWNLOAD REQUIRED NLTK RESOURCES
# ==========================================================

@st.cache_resource
def setup_nltk():

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet")
    ]

    for resource_path, resource_name in resources:

        try:
            nltk.data.find(resource_path)

        except LookupError:
            nltk.download(resource_name, quiet=True)


setup_nltk()


# ==========================================================
# LOAD MODEL AND TF-IDF VECTORIZER
# ==========================================================

@st.cache_resource
def load_artifacts():

    base_path = Path(__file__).parent

    model_path = base_path / "final_sentiment_model.pkl"

    vectorizer_path = (
        base_path / "final_tfidf_vectorizer.pkl"
    )

    model = joblib.load(model_path)

    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


try:

    model, tfidf_vectorizer = load_artifacts()

except Exception as e:

    st.error(
        "Unable to load the model or TF-IDF vectorizer."
    )

    st.exception(e)

    st.stop()


# ==========================================================
# PREPROCESSING SETUP
# ==========================================================

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))


# ==========================================================
# TEXT PREPROCESSING FUNCTION
# ==========================================================

def preprocess_text(text):

    # Convert input to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Keep English characters,
    # Hindi / Devanagari characters and spaces
    text = re.sub(
        r"[^a-zA-Z\u0900-\u097F\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove English stopwords
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    # Convert tokens back to text
    processed_text = " ".join(tokens)

    return processed_text


# ==========================================================
# LANGUAGE TYPE DETECTION
# ==========================================================

def detect_language_type(text):

    text = str(text)

    # Check for Hindi / Devanagari characters
    if re.search(
        r"[\u0900-\u097F]",
        text
    ):
        return "Hindi Script"

    # Common Hinglish words
    hinglish_words = [
        "acha",
        "achha",
        "achcha",
        "bahut",
        "bekar",
        "bekaar",
        "badiya",
        "badhiya",
        "mast",
        "theek",
        "thik",
        "hai",
        "hain",
        "nahi",
        "nahin",
        "kyu",
        "kyun",
        "kya",
        "ye",
        "yeh",
        "mera",
        "meri",
        "mere",
        "iska",
        "iski",
        "iske",
        "bahar",
        "andar",
        "paise",
        "paisa",
        "waste",
        "kar",
        "karo",
        "kiya",
        "liye",
        "wala",
        "wali",
        "wala",
        "milta",
        "mila",
        "mujhe",
        "aap",
        "tum",
        "bhai",
        "bahut hi",
        "sahi"
    ]

    text_lower = text.lower()

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text_lower
    )

    if any(
        word in hinglish_words
        for word in words
    ):
        return "Possible Hinglish"

    return "Other / Likely English"


# ==========================================================
# SENTIMENT PREDICTION FUNCTION
# ==========================================================

def predict_sentiment(review):

    # Apply preprocessing
    processed_review = preprocess_text(review)

    # Convert to TF-IDF features
    review_tfidf = tfidf_vectorizer.transform(
        [processed_review]
    )

    # Predict sentiment
    prediction = model.predict(
        review_tfidf
    )[0]

    return prediction


# ==========================================================
# FUNCTION FOR BATCH PREDICTION
# ==========================================================

def predict_batch(reviews):

    processed_reviews = [
        preprocess_text(review)
        for review in reviews
    ]

    reviews_tfidf = tfidf_vectorizer.transform(
        processed_reviews
    )

    predictions = model.predict(
        reviews_tfidf
    )

    return predictions


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown(
    "# 💬 Sentiment Analysis"
)

st.sidebar.markdown(
    "Customer Review Classification"
)

st.sidebar.markdown("---")


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "💬 Single Prediction",
        "📂 Batch Prediction",
        "📊 Model Information"
    ]
)


st.sidebar.markdown("---")


st.sidebar.success(
    """
    **Final Model**

    ✔ Balanced Linear SVM

    **Accuracy**

    78.12%

    **Text Representation**

    TF-IDF (1,2)

    **Sentiments**

    Positive, Negative, Neutral
    """
)


# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":

    st.title(
        "💬 Customer Review Sentiment Analysis"
    )

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:

        st.markdown(
            """
            ### About the Project

            This application analyzes customer reviews and predicts
            whether the sentiment is **Positive, Negative, or Neutral**.

            The application uses a trained Machine Learning model
            based on customer review data.

            ### Application Features

            ✅ Single Review Sentiment Prediction

            ✅ English, Hinglish and Hindi Script Input Support

            ✅ Batch Prediction using CSV Files

            ✅ Sentiment Distribution Visualization

            ✅ Download Prediction Results

            ✅ Built using Streamlit
            """
        )

    with col2:

        st.metric(
            "Model Accuracy",
            "78.12%"
        )

        st.metric(
            "Macro F1-Score",
            "0.65"
        )

        st.metric(
            "Sentiment Classes",
            "3"
        )

    st.markdown("---")

    st.subheader(
        "How the Application Works"
    )

    workflow_data = pd.DataFrame(
        {
            "Step": [
                "1",
                "2",
                "3",
                "4",
                "5"
            ],
            "Process": [
                "User enters a customer review",
                "Review goes through text preprocessing",
                "Saved TF-IDF vectorizer converts text into numerical features",
                "Balanced Linear SVM analyzes the features",
                "Predicted sentiment is displayed"
            ]
        }
    )

    st.dataframe(
        workflow_data,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Use the sidebar to analyze a single review "
        "or upload multiple reviews for batch prediction."
    )


# ==========================================================
# SINGLE REVIEW PREDICTION
# ==========================================================

elif page == "💬 Single Prediction":

    st.title(
        "💬 Single Review Sentiment Prediction"
    )

    st.markdown(
        """
        Enter a customer review below.

        The application can accept English, Hinglish,
        and Hindi-script text.
        """
    )

    st.markdown("---")

    review = st.text_area(
        "Enter Customer Review",
        placeholder=(
            "Example: This phone is very good "
            "and the camera quality is excellent."
        ),
        height=180
    )

    st.caption(
        "Examples: English, Hinglish, or Hindi-script reviews."
    )

    predict_button = st.button(
        "Analyze Sentiment"
    )

    if predict_button:

        if not review.strip():

            st.warning(
                "Please enter a customer review before analyzing."
            )

        else:

            # Detect language type
            language_type = detect_language_type(
                review
            )

            # Make prediction
            prediction = predict_sentiment(
                review
            )

            st.markdown("---")

            st.subheader(
                "Analysis Result"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Detected Review Type",
                    language_type
                )

            with col2:

                st.metric(
                    "Predicted Sentiment",
                    prediction
                )

            # Display sentiment result

            if prediction == "Positive":

                st.success(
                    "😊 Positive Sentiment Detected"
                )

            elif prediction == "Negative":

                st.error(
                    "😞 Negative Sentiment Detected"
                )

            else:

                st.info(
                    "😐 Neutral Sentiment Detected"
                )

            with st.expander(
                "View Processed Review"
            ):

                processed_review = preprocess_text(
                    review
                )

                st.write(
                    processed_review
                )


# ==========================================================
# BATCH PREDICTION
# ==========================================================

elif page == "📂 Batch Prediction":

    st.title(
        "📂 Batch Review Sentiment Analysis"
    )

    st.markdown(
        """
        Upload a CSV file containing customer reviews.
        The application will predict the sentiment for
        each review.
        """
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    st.info(
        "The CSV file must contain a column named "
        "`review`."
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(
                uploaded_file
            )

            st.subheader(
                "Uploaded Data Preview"
            )

            st.dataframe(
                batch_df.head(),
                use_container_width=True
            )

            # Check required column

            if "review" not in batch_df.columns:

                st.error(
                    "The uploaded CSV must contain "
                    "a column named 'review'."
                )

            else:

                # Remove missing / empty reviews

                valid_mask = (
                    batch_df["review"]
                    .notna()
                    &
                    batch_df["review"]
                    .astype(str)
                    .str.strip()
                    .ne("")
                )

                invalid_count = (
                    (~valid_mask).sum()
                )

                prediction_df = batch_df.loc[
                    valid_mask
                ].copy()

                if prediction_df.empty:

                    st.warning(
                        "No valid reviews were found "
                        "in the uploaded CSV."
                    )

                else:

                    if invalid_count > 0:

                        st.warning(
                            f"{invalid_count} empty or missing "
                            "review(s) were excluded from prediction."
                        )

                    if st.button(
                        "Analyze All Reviews"
                    ):

                        reviews = (
                            prediction_df["review"]
                            .astype(str)
                            .tolist()
                        )

                        with st.spinner(
                            "Analyzing customer reviews..."
                        ):

                            # Detect review type
                            prediction_df[
                                "Detected Review Type"
                            ] = [
                                detect_language_type(review)
                                for review in reviews
                            ]

                            # Predict sentiment
                            prediction_df[
                                "Predicted Sentiment"
                            ] = predict_batch(
                                reviews
                            )

                        st.success(
                            "Batch prediction completed successfully."
                        )

                        st.markdown("---")

                        st.subheader(
                            "Prediction Results"
                        )

                        st.dataframe(
                            prediction_df,
                            use_container_width=True
                        )

                        # Sentiment distribution

                        st.markdown("---")

                        st.subheader(
                            "Sentiment Distribution"
                        )

                        sentiment_counts = (
                            prediction_df[
                                "Predicted Sentiment"
                            ]
                            .value_counts()
                            .reset_index()
                        )

                        sentiment_counts.columns = [
                            "Sentiment",
                            "Count"
                        ]

                        fig = px.bar(
                            sentiment_counts,
                            x="Sentiment",
                            y="Count",
                            text="Count",
                            title=(
                                "Distribution of Predicted Sentiments"
                            )
                        )

                        fig.update_traces(
                            textposition="outside"
                        )

                        fig.update_layout(
                            xaxis_title="Sentiment",
                            yaxis_title="Number of Reviews"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        # Review type distribution

                        st.subheader(
                            "Review Type Distribution"
                        )

                        language_counts = (
                            prediction_df[
                                "Detected Review Type"
                            ]
                            .value_counts()
                            .reset_index()
                        )

                        language_counts.columns = [
                            "Review Type",
                            "Count"
                        ]

                        language_fig = px.pie(
                            language_counts,
                            names="Review Type",
                            values="Count",
                            title=(
                                "Detected Review Type Distribution"
                            )
                        )

                        st.plotly_chart(
                            language_fig,
                            use_container_width=True
                        )

                        # Download results

                        st.markdown("---")

                        csv_data = (
                            prediction_df
                            .to_csv(index=False)
                            .encode("utf-8")
                        )

                        st.download_button(
                            label="Download Prediction Results",
                            data=csv_data,
                            file_name=(
                                "sentiment_prediction_results.csv"
                            ),
                            mime="text/csv"
                        )

        except Exception as e:

            st.error(
                "An error occurred while processing "
                "the uploaded CSV file."
            )

            st.exception(e)


# ==========================================================
# MODEL INFORMATION PAGE
# ==========================================================

elif page == "📊 Model Information":

    st.title(
        "📊 Model Information"
    )

    st.markdown("---")

    st.subheader(
        "Final Model"
    )

    st.write(
        """
        The final model selected for this project is
        **Balanced Linear Support Vector Machine (Linear SVM)**.
        """
    )

    model_info = pd.DataFrame(
        {
            "Component": [
                "Final Model",
                "Text Representation",
                "N-gram Range",
                "Class Weight",
                "Accuracy",
                "Macro F1-Score",
                "Weighted F1-Score"
            ],
            "Details": [
                "Balanced Linear SVM",
                "TF-IDF",
                "Unigrams + Bigrams (1,2)",
                "Balanced",
                "78.12%",
                "0.65",
                "0.77"
            ]
        }
    )

    st.dataframe(
        model_info,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader(
        "Text Preprocessing"
    )

    preprocessing_steps = pd.DataFrame(
        {
            "Step": [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6"
            ],
            "Process": [
                "Convert text to lowercase",
                "Remove URLs",
                "Remove unwanted characters",
                "Remove extra spaces",
                "Tokenization and English stopword removal",
                "Lemmatization"
            ]
        }
    )

    st.dataframe(
        preprocessing_steps,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader(
        "Model Performance Comparison"
    )

    comparison_df = pd.DataFrame(
        {
            "Model": [
                "Multinomial Naive Bayes",
                "Logistic Regression",
                "Linear SVM",
                "Random Forest",
                "Balanced Linear SVM"
            ],
            "Accuracy (%)": [
                76.74,
                78.12,
                78.47,
                77.43,
                78.12
            ],
            "Macro F1-Score": [
                0.55,
                0.59,
                0.64,
                0.57,
                0.65
            ]
        }
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    comparison_fig = px.bar(
        comparison_df,
        x="Model",
        y="Accuracy (%)",
        text="Accuracy (%)",
        title="Model Accuracy Comparison"
    )

    comparison_fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    comparison_fig.update_layout(
        xaxis_tickangle=-30
    )

    st.plotly_chart(
        comparison_fig,
        use_container_width=True
    )

    st.info(
        """
        The Balanced Linear SVM was selected as the final model
        because it provided slightly better balanced performance
        across all three sentiment classes, especially for the
        Neutral class.
        """
    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    """
    <div class="footer">
        Customer Review Sentiment Analysis |
        Built using Streamlit, TF-IDF and Linear SVM
    </div>
    """,
    unsafe_allow_html=True
)