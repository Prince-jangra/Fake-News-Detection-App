import streamlit as st
import joblib
import os

# Load the trained model and vectorizer
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function to make predictions
def predict_news(news_text):
    transformed_text = vectorizer.transform([news_text])
    prediction = model.predict(transformed_text)
    return "Real News" if prediction[0] == 1 else "Fake News"

# Adjust text positioning
st.title("Fake News Detection")

st.markdown(
    """
    ### Welcome to the Fake News Detection App!
    Identify whether a news article is **Real** or **Fake** in seconds.
    Paste the article text below and click **Predict News**.
    """
)

# Sidebar Information
st.sidebar.title("About this App")
st.sidebar.markdown(
    """
    This app uses a machine learning model to detect whether a news article is real or fake.
    
    **Steps**:
    1. Paste a news article.
    2. Click "Predict News".
    3. The result will be shown below.
    """
)

# Main section (Shifted Upwards)
st.header("Enter News Article for Prediction")
news_input = st.text_area("Paste the news article here:", height=130)

if st.button("Predict News"):
    if news_input:
        with st.spinner("Processing your news article..."):
            result = predict_news(news_input)
        if result == "Real News":
            st.success(f"The article is **Real News**!")
        else:
            st.error(f"The article is **Fake News**!")
    else:
        st.warning("Please enter some news article text before predicting.")

# Model file existence check (for debugging)
print("Model Exists:", os.path.exists("best_model.pkl"))
print("Vectorizer Exists:", os.path.exists("vectorizer.pkl"))

if os.path.exists("best_model.pkl") and os.path.exists("vectorizer.pkl"):
    model = joblib.load("best_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print("Model Type:", type(model))
    print("Vectorizer Type:", type(vectorizer))
else:
    print("Model or Vectorizer file not found!")

import os
import joblib

# Verify if the model and vectorizer files exist
print("Model Exists:", os.path.exists("best_model.pkl"))
print("Vectorizer Exists:", os.path.exists("vectorizer.pkl"))

# Load the model and vectorizer
if os.path.exists("best_model.pkl") and os.path.exists("vectorizer.pkl"):
    model = joblib.load("best_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    # Check the model and vectorizer types to ensure they've loaded correctly
    print("Model Type:", type(model))
    print("Vectorizer Type:", type(vectorizer))
else:
    print("Model or Vectorizer file not found!")

