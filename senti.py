import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import pandas as pd
import matplotlib.pyplot as plt


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


# Perform sentiment analysis
@st.cache_data
def perform_sentiment_analysis(texts, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    results = []
    labels = []
    for text in texts:
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors="pt")
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        result = [
            (config.id2label[ranking[i]], np.round(float(scores[ranking[i]]), 4))
            for i in range(scores.shape[0])
        ]
        results.append(result)
        labels.append(config.id2label[ranking[0]])
    return results, labels


# Streamlit app
def main():
    st.title("Employee Pulse")
    st.write("Analyze feedback data by department.")

    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        departments = ["None"] + df["Department"].unique().tolist()
        selected_department = st.sidebar.selectbox(
            "Select a department for detailed analysis", departments
        )

        if selected_department == "None":
            # Overall dataset analysis
            st.write("## Overall Dataset Analysis")
            all_texts = df["Feedback"].tolist()
            results, labels = perform_sentiment_analysis(all_texts, MODEL)

            positive = labels.count("positive")
            negative = labels.count("negative")
            neutral = labels.count("neutral")

            st.write(f"Positive: {positive}, Negative: {negative}, Neutral: {neutral}")

            # Plotting pie chart
            fig, ax = plt.subplots()
            ax.pie(
                [positive, negative, neutral],
                labels=["Positive", "Negative", "Neutral"],
                autopct="%1.1f%%",
            )
            ax.axis("equal")
            st.pyplot(fig)

            st.write("### Detailed Text Analysis")
            for text, result in zip(all_texts, results):
                st.write(f"Text: {text}")
                for label, score in result:
                    st.write(f"{label}: {score}")
                st.write("---")
        else:
            # Department-wise analysis
            st.write(f"## Analysis for {selected_department} Department")
            department_texts = df[df["Department"] == selected_department][
                "Feedback"
            ].tolist()
            results, labels = perform_sentiment_analysis(department_texts, MODEL)

            positive = labels.count("positive")
            negative = labels.count("negative")
            neutral = labels.count("neutral")

            st.write(f"Positive: {positive}, Negative: {negative}, Neutral: {neutral}")

            # Plotting pie chart for selected department
            fig, ax = plt.subplots()
            ax.pie(
                [positive, negative, neutral],
                labels=["Positive", "Negative", "Neutral"],
                autopct="%1.1f%%",
            )
            ax.axis("equal")
            st.pyplot(fig)

            st.write("### Detailed Text Analysis for Selected Department")
            for text, result in zip(department_texts, results):
                st.write(f"Text: {text}")
                for label, score in result:
                    st.write(f"{label}: {score}")
                st.write("---")


if __name__ == "__main__":
    main()
