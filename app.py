import joblib
model = joblib.load("fake_news_detection_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("fake_news_detection_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit app UI
st.title("📰 Fake News Detection App")
st.write("Enter a news article text, and the model will predict whether it's Fake or Real.")

# Text input
user_input = st.text_area("Enter news text here:")

if st.button("Predict"):
    if user_input:
        # Transform the input using the vectorizer
        transformed_text = vectorizer.transform([user_input])

        # Make prediction
        prediction = model.predict(transformed_text)

        # Display result
        result = "🟢 Real News" if prediction[0] == 0 else "🔴 Fake News"
        st.subheader(result)
    else:
        st.warning("Please enter some text to analyze.")
