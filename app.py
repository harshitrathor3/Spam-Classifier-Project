import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Define custom styles
PRIMARY_COLOR = "#2E8B57"  # Sea Green
SECONDARY_COLOR = "#FF6347"  # Tomato
BACKGROUND_COLOR = "#E0E0E0"  # Light Gray
FONT_FAMILY = "Arial, sans-serif"
TEXT_COLOR = "#000000"  # Black
HEADER_COLOR = "#1E1E1E"  # Dark Gray

# Apply global styles
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BACKGROUND_COLOR};
        font-family: {FONT_FAMILY};
        color: {TEXT_COLOR};
    }}
    .stButton > button {{
        background-color: {PRIMARY_COLOR};
        color: white;
        font-size: 1.2em;
        padding: 10px 20px;
        border-radius: 10px;
        transition: background-color 0.3s;
    }}
    .stButton > button:hover {{
        background-color: {SECONDARY_COLOR};
    }}
    .stTextArea {{
        border-color: {PRIMARY_COLOR};
        border-width: 2px;
        border-radius: 10px;
    }}
    .result-box {{
        border: 2px solid {PRIMARY_COLOR};
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        font-size: 1.5em;
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
    }}
    .stTitle, .stHeader {{
        color: {TEXT_COLOR};
        background-color: {HEADER_COLOR};
        padding: 10px;
        border-radius: 10px;
    }}
    .navbar {{
        background-color: {HEADER_COLOR};
        padding: 10px;
        border-radius: 10px;
    }}
    .navbar .nav-item {{
        padding: 10px 20px;
    }}
    .navbar .nav-item .nav-link {{
        color: {TEXT_COLOR};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Define the PorterStemmer instance
ps = PorterStemmer()

# Define a function to transform the text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [word for word in text if word.isalnum()]

    y = [ps.stem(word) for word in y if word not in stopwords.words("english")]

    return " ".join(y)

# # Load the pre-trained TF-IDF vectorizer and model
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Define the home page
def home():
    st.title("Welcome to Spam Detector App")
    st.subheader("**This app helps you detect whether an email or SMS is spam or not.This spam detector utilizes several machine learning algorithms to predict spam.Models include Support Vector Classifier (SVC), Voting Classifier, and Naive Bayes.Trained on a diverse dataset, they accurately classify messages as spam or not spam.**")
   

# Define the spam detector page
def spam_detector():
    st.title("🛑 Spam Detector")
    st.subheader("Predict whether an email or SMS is spam or not.")

    # Define the text area for input
    input_sms = st.text_area("Enter your message", height=150)

    # Predict button with interactive effect
    if st.button("Detect"):
        # Preprocess the input text
        transformed_sms = transform_text(input_sms)
        
        # # Vectorize the transformed text
        vector_input = tfidf.transform([transformed_sms])
        
        # Predict the result
        result = model.predict(vector_input)[0]
        
        # Display the result with an updated layout and styling
        if result == 1:
            st.markdown(
                "<div class='result-box'>🛑 Spam Detected 🛑</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='result-box'>✅ Not Spam ✅</div>",
                unsafe_allow_html=True,
            )

# Render the appropriate page based on selection
page = st.sidebar.radio("Select a page:", ("Home", "Spam Detector"))

if page == "Home":
    home()
elif page == "Spam Detector":
    spam_detector()
