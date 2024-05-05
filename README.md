# Spam-Classifier-Project

# User Manual for Spam Detector App

## Introduction

The Spam Detector App is a machine learning-based application that can detect whether an email or SMS message is spam or not. The app utilizes several machine learning algorithms, including Support Vector Classifier (SVC), Voting Classifier, and Naive Bayes, to accurately classify messages as spam or not spam.

## Installation

To run the Spam Detector App, you need to have the following dependencies installed:

- Python 3.x
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- NLTK
- Seaborn
- Matplotlib
- WordCloud

You can install the required Python packages using the following command:

```
pip install streamlit numpy pandas scikit-learn nltk seaborn matplotlib wordcloud
```

Additionally, you need to download the NLTK data by running the following commands in your Python interpreter:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

Running the App

1. Clone or download the project repository to your local machine.
2. Navigate to the project directory in your terminal or command prompt.
3. Run the following command to start the Streamlit app:
```
streamlit run app.py
```

4. The app will open in your default web browser. If it doesn't open automatically, you can access it by copying the URL provided in the terminal and pasting it into your web browser.


## Using the App
The Spam Detector App has two main pages: "Home" and "Spam Detector".
## Home Page
The Home page provides a brief introduction to the Spam Detector App and its functionality.
## Spam Detector Page
The Spam Detector page allows you to enter an email or SMS message in the provided text area and click the "Detect" button to determine whether the message is spam or not.
After clicking the "Detect" button, the app will preprocess the input text, vectorize it using the pre-trained TF-IDF vectorizer, and then pass it through the pre-trained machine learning model to predict the result.

The result will be displayed in a styled box with either "🛑 Spam Detected 🛑" or "✅ Not Spam ✅" message.

## Customization

If you want to customize the app's appearance or behavior, you can modify the code in the app.py file.

- To change the color scheme, update the PRIMARY_COLOR, SECONDARY_COLOR, BACKGROUND_COLOR, TEXT_COLOR, and HEADER_COLOR variables at the top of the app.py file.
- To modify the text preprocessing steps, update the transform_text function in the app.py file.
- To use a different machine learning model or vectorizer, update the tfidf and model variables in the app.py file with your own pre-trained models. You can train new models by modifying the spam_detection_final.ipynb file.

## Conclusion
The Spam Detector App provides a user-friendly interface for detecting spam messages using machine learning algorithms. By following the instructions in this user manual, you can easily run and use the app on your local machine.
