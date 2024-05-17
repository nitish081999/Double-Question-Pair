import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import dask
from dask import dataframe as dd
from nltk.corpus import stopwords
import distance
from  fuzzywuzzy import fuzz
import gensim
from gensim.utils import simple_preprocess
from nltk import sent_tokenize
from tqdm import tqdm
import joblib

def preprocess(q):
    q = str(q).lower().strip()

    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')

    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')

    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()

    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    return q

def common_words(row):
    w1=set(map(lambda word:word.lower().strip(),row['question1'].split(' ')))
    w2=set(map(lambda word:word.lower().strip(),row['question2'].split(' ')))
    return len(w1&w2)

def total_words(row):
    w1=set(map(lambda word:word.lower().strip(),row['question1'].split(' ')))
    w2=set(map(lambda word:word.lower().strip(),row['question2'].split(' ')))
    return len(w1)+len(w2)


def fetch_token_features(row):
    q1 = row['question1']
    q2 = row['question2']

    SAFE_DIV = 0.0001

    STOP_WORDS = stopwords.words('english')

    token_features = [0.0] * 8

    # converting sentence into Tokens
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))

    common_stop_count = len(q1_stops.intersection(q2_stops))

    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens) + SAFE_DIV))
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens) + SAFE_DIV))

    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])

    token_features[7] = int(q1_tokens[0] == q1_tokens[0])

    return token_features


def fetch_length_features(row):
    q1 = row['question1']
    q2 = row['question2']

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    len_features = [0] * 3  # Initialize length features list

    # Feature 1: Absolute length difference
    len_features[0] = abs(len(q1_tokens) - len(q2_tokens))

    # Feature 2: Mean length
    len_features[1] = abs(len(q1_tokens) + len(q2_tokens)) / 2

    # Feature 3: Longest common substring ratio
    if len(q1_tokens) > 0 and len(q2_tokens) > 0:
        strs = list(distance.lcsubstrings(q1, q2))
        if strs:  # Check if strs is not empty
            len_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)

    return len_features


def fetch_fuzzy_features(row):
    q1 = row['question1']
    q2 = row['question2']

    fuzzy_features = [0.0] * 4

    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features





model_word2vec=joblib.load('word2vec_model.joblib')

def document_vector(doc):
    doc=[word for word in doc.split() if word in model_word2vec.wv.index_to_key]
    if doc:
        return np.mean(model_word2vec.wv[doc],axis=0)
    else:
        return np.zeros(model_word2vec.vector_size)

st.title('Is Same Questions ?')

user_input1 = st.text_input('Enter Question 1')
user_input2 = st.text_input('Enter Question 2')

clicked = st.button('Predict')


def input_pipeline(que1, que2):
    temp = [{'question1': que1, 'question2': que2}]
    temp = pd.DataFrame(temp)


    temp['question1'] = temp['question1'].apply(preprocess)
    temp['question2'] = temp['question2'].apply(preprocess)

    temp['q1_len'] = temp['question1'].str.len()
    temp['q2_len'] = temp['question2'].str.len()

    temp['q1_words'] = temp['question1'].apply(lambda row: len(row.split(' ')))
    temp['q2_words'] = temp['question2'].apply(lambda row: len(row.split(' ')))

    temp['total_words'] = temp.apply(total_words, axis=1)

    temp['common_words'] = temp.apply(common_words, axis=1)

    temp['word_share'] = temp['common_words'] / temp['total_words']

    token_features = temp.apply(fetch_token_features, axis=1)
    temp['cwc_min'] = list(map(lambda x: x[0], token_features))
    temp['cwc_max'] = list(map(lambda x: x[1], token_features))
    temp['csc_min'] = list(map(lambda x: x[2], token_features))
    temp['csc_max'] = list(map(lambda x: x[3], token_features))
    temp['ctc_min'] = list(map(lambda x: x[4], token_features))
    temp['ctc_max'] = list(map(lambda x: x[5], token_features))
    temp['last_word_eq'] = list(map(lambda x: x[6], token_features))
    temp['first_word_eq'] = list(map(lambda x: x[7], token_features))

    length_features = temp.apply(fetch_length_features, axis=1)

    temp['abs_len_diff'] = list(map(lambda x: x[0], length_features))
    temp['mean_length'] = list(map(lambda x: x[1], length_features))
    temp['longest_substr_ratio'] = list(map(lambda x: x[2], length_features))

    fuzzy_features = temp.apply(fetch_fuzzy_features, axis=1)
    temp['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_features))
    temp['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_features))
    temp['token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
    temp['token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))

    temp_df = temp[['question1', 'question2']]

    temp.drop(columns=['question1', 'question2'], inplace=True)

    que_list = list(temp_df['question1']) + list(temp_df['question2'])

    temp_questions = []


    for que in que_list:
        raw_sent = sent_tokenize(que)
        for sent in raw_sent:
            temp_questions.append(simple_preprocess(sent))

    input_que1 = []
    input_que2 = []

    for doc in tqdm(temp_df['question1'].values):
        input_que1.append(document_vector(doc))

    for doc in tqdm(temp_df['question2'].values):
        input_que2.append(document_vector(doc))

    input_que1 = np.array(input_que1)
    input_que2 = np.array(input_que2)

    return np.hstack((temp.values, input_que1, input_que2))


def print_function(user_input1,user_input2):
    input_questions = input_pipeline(user_input1, user_input2)
    ans = model.predict(input_questions)
    if ans == 1:
        st.write('Same Questions')
    else:
        st.write('Different Questions')



model = pickle.load(open('model.pkl', 'rb'))

if clicked:
    print_function(user_input1,user_input2)




