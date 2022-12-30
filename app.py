import streamlit as st
from PIL import Image
import pickle
import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords


# img_file_buffer = st.camera_input("Take a picture")

# if img_file_buffer is not None:
#     # To read image file buffer as bytes:
#     bytes_data = img_file_buffer.getvalue()
#     # Check the type of bytes_data:
#     # Should output: <class 'bytes'>
#     # st.write(type(bytes_data))

st.subheader("Predict the Amazon Review Score")
# st.text_input("Input your review:")
review = st.text_input("Review Summary:")
review_desc = st.text_area("Review Description:")
stop_words = stopwords.words("english") + ["br", "html", "www", "k", "http"]

def set_unique_words(lem_text):
    res = lambda lem_text: ' '.join(set(row_word for row_word in lem_text.split(" ") if row_word not in stop_words))
    return res(lem_text)

def load_module(lem_text):
    if int(len(lem_text)) != 1:
        # st.write(len(lem_text))
        rf_bow_amazon_rev = pickle.load(open("rf_amazon_rev_bow.sav", "rb"))
        rf_amazon_rev = pickle.load(open("rf_amazon_rev.sav", "rb"))
        X_test_ = rf_bow_amazon_rev.transform([lem_text])  
        score = rf_amazon_rev.predict(X_test_.toarray().reshape(1, -1))
        st.write("Predicted Review Score:  {0}".format(score[0]))
        # st.write("★ "*score[0])
        # st.write("⭐️ "*score[0], )
        # st.write("⭑ "*score[0], "⭒ "*(5-score[0]))
        st.write("★ "*score[0], "☆ "*(5-score[0]))
        

def preprocess(final_text):
    final_text = list(map(lambda x:x.replace(":)", "good "),  final_text.split(" ")))
    final_text = list(map(lambda x:x.replace(":(", "bad "),  final_text))
    final_text_re = re.sub("[^a-zA-Z]", " ", str(final_text).lower())
    lemmatize = WordNetLemmatizer() # stemmer = PorterStemmer()
    lem_text = [lemmatize.lemmatize(x.lower()) for x in final_text_re.split(" ") if x not in stop_words]
    lem_text = [lemmatize.lemmatize(x) for x in lem_text if x!=""]
    lem_text = " ".join([x for x in lem_text])
    return lem_text

if st.button("Submit Review !!", help="Predict the \"Amazon Product Review\" "):
    final_text = review + " " + review_desc  
    # st.write("No of chars: {0}".format( int(len(final_text))-1))
    if int(len(final_text)) == 1:
        st.write("Please input the Review")
    else:
        lem_text =  preprocess(final_text)
        lem_text = set_unique_words(lem_text)
        # st.write(lem_text)
        load_module(lem_text)




