"""
Functions for preprocessing to clean data, create new features and appling NLP-Steps to the text
"""

import re
import string

import nltk
import spacy
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob
from langdetect import DetectorFactory, detect
from nltk.corpus import stopwords

nlp = spacy.load('en_core_web_sm')
nltk.download('stopwords')
stopwords = stopwords.words('english')
PUNCTUATIONS = string.punctuation


def clean_data(features, target):
    """
    Function for data cleaning
    """
    # drop unused column
    features = features.drop(columns='Unnamed: 0')
    # Reverse labels into: fake=1, real=0
    target = 1 - target
    # Fill Nan-values with an empty string
    features[['title', 'text']] = features[['title', 'text']].fillna('')

    return features, target


def get_sentiment(text):
    """
    Function for calculation sentiment scores
    """
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity


def avg_word_length(text):
    """
    Function for calculating the average word length
    """
    words = text.split()
    if len(words) == 0:
        return 0
    return sum(len(word) for word in words) / len(words)


def sentence_count(text):
    """
    Function for calculating the number of sentences
    """
    sentences = re.split(r'[.!?]', text)  # Sentence splitting for '.', '!', '?'
    sentences = [s for s in sentences if s.strip()]  # Removes empty sentences
    return len(sentences)


def special_char_count(text):
    """
    Function for counting special characters
    """
    return len(re.findall(r'[^a-zA-Z0-9\s]', text))  # Finds all special characters


DetectorFactory.seed = 42


def detect_language(text):
    """
    Function to detect language in text column
    """
    try:
        return detect(text)
    except:
        return 'unknown'


def prepare_features(df):
    """
    Create new features
    - title_text: concatenate columns title and text
    - text_word_count: # of words in text column
    - title_word_count: # of words in title column
    - text_unique_words: # of uniqe words in text column
    - text_char_count: # of characters in text column
    - title_char_count: # of characters in title column
    - avg_word_length: average word length in text column
    - sentence_count: # of sentences in text column
    - special_char_count: # of special characters in text column
    - language: estimated language for text column
    - sentiment: calculate sentiment scores

    Args:
        DataFrame wit columns 'title' and 'text'

    Returns:
        DataFrame with new columns mentioned above.
        New column 'title_text' will be needed for function apply_text_cleaner to apply NLP-Steps to this column
        After creating new features columns 'title' and 'text' will be removed.

    """
    df = df.copy()
    df['title_text'] = df['title'] + ' ' + df['text']

    df['text_word_count'] = df["text"].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) else 0
    )
    df['title_word_count'] = df["title"].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) else 0
    )

    df['text_unique_words'] = df["text"].apply(
        lambda x: len(set(x.lower().split())) if pd.notnull(x) else 0
    )

    df['text_char_count'] = df["text"].apply(
        lambda x: len(x) - x.count(' ') if pd.notnull(x) else 0
    )
    df['title_char_count'] = df["title"].apply(
        lambda x: len(x) - x.count(' ') if pd.notnull(x) else 0
    )

    # Applying the functions to the text column
    df['avg_word_length'] = df['text'].apply(lambda x: avg_word_length(str(x)))
    df['sentence_count'] = df['text'].apply(lambda x: sentence_count(str(x)))
    df['special_char_count'] = df['text'].apply(lambda x: special_char_count(str(x)))
    # df['language'] = df['text'].astype(str).apply(detect_language)
    df['sentiment'] = df['text'].apply(get_sentiment)

    df = df.drop(['title', 'text'], axis=1)

    return df


def text_cleaner(sentence):
    """Clean the text using NLP-Steps.

    Steps include: Lemmatization, removing stop words, removing punctuations

    Args:
        sentence (str): The uncleaned text.

    Returns:
        str: The cleaned text.

    """

    # Create the Doc object named `text` from `sentence` using `nlp()`
    doc = nlp(sentence)
    # Lemmatization
    lemma_token = [token.lemma_ for token in doc if token.pos_ != 'PRON']
    # Remove stop words and converting tokens to lowercase
    no_stopwords_lemma_token = [
        token.lower() for token in lemma_token if token not in stopwords
    ]
    # Remove punctuations
    clean_doc = [
        token for token in no_stopwords_lemma_token if token not in PUNCTUATIONS
    ]
    # Use the `.join` method on `text` to convert string
    joined_clean_doc = " ".join(clean_doc)
    # Use `re.sub()` to substitute multiple spaces or dots`[\.\s]+` to single space `' '
    sub_doc = re.sub(r'[\.\s]+', ' ', joined_clean_doc)
    # Use `re.sub()` to substitute ■ to an empty string `' '
    final_doc = re.sub(r'[\■🚨]+', '', sub_doc)
    return final_doc


def apply_text_cleaner(df, column):
    """
    Clean the text using NLP-Steps which are defined in the function text_cleaner.

    Steps include: Lemmatization, removing stop words, removing punctuations

    Args:
        sentence (str): The uncleaned text.

    Returns:
        str: The cleaned text.
    """
    # Progress bar
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Progress'):
        results.append(text_cleaner(row[column]))

    df['title_text_clean'] = results

    df = df.drop(['title_text'], axis=1)

    return df
