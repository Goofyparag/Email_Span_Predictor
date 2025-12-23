import streamlit as st
import nltk
import pickle
import string
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


nltk.data.path.append(os.path.expanduser("~/.nltk_data"))

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    filtered_words = []
    for word in text:
        if word.isalnum():
            filtered_words.append(word)

    filtered_words = [
        ps.stem(word)
        for word in filtered_words
        if word not in stopwords.words("english") and word not in string.punctuation
    ]

    return " ".join(filtered_words)



with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📩 SMS / Message Spam Classifier")

input_sms = st.text_area("Enter the message:", height=150)

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed_sms = transform_text(input_sms)
        vectorized_sms = vectorizer.transform([transformed_sms])
        prediction = model.predict(vectorized_sms)[0]

        if prediction == 1:
            st.error(" SPAM Message Detected!")
        else:
            st.success("Not Spam (Ham).")
