import streamlit as st
import joblib

# Load the trained model
model = joblib.load("email_classifier_model.pkl")

st.title("📧 Email Authenticity Checker")
st.markdown("Check if an email is **legit** or **phishing** using AI.")

# Input fields
subject = st.text_input("Email Subject")
sender = st.text_input("Sender Email")
body = st.text_area("Email Body", height=200)

if st.button("Check Email"):
    # Combine text for model input
    combined_text = subject + " " + sender + " " + body

    # Predict
    prediction = model.predict([combined_text])[0]
    probability = model.predict_proba([combined_text])[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ This email is likely **phishing** ({round(probability * 100, 2)}% confidence)")
    else:
        st.success(f"✅ This email appears **legit** ({round(probability * 100, 2)}% confidence)")
