import pandas as pd
import string
import re

import spacy
import nltk

from tqdm import tqdm
from nltk.corpus import stopwords

from langdetect import detect, DetectorFactory

nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
stopWords = stopwords.words('english')
punctuations = string.punctuation

# Function for data cleaning
def clean_data(features, target):
    # drop unused column
    features = features.drop(columns='Unnamed: 0')
    # Reverse labels into: fake=1, real=0
    target = 1 - target
    # Fill Nan-values with an empty string
    features[['title', 'text']] = features[['title', 'text']].fillna("")

    return features, target

# Function for calculating the average word length
def avg_word_length(text):
    words = text.split()
    if len(words) == 0:
        return 0
    return sum(len(word) for word in words) / len(words)

# Function for calculating the number of sentences
def sentence_count(text):
    sentences = re.split(r'[.!?]', text)  # Sentence splitting for '.', '!', '?'
    sentences = [s for s in sentences if s.strip()]  # Removes empty sentences
    return len(sentences)

# Function for counting special characters
def special_char_count(text):
    return len(re.findall(r'[^a-zA-Z0-9\s]', text))  # Finds all special characters

DetectorFactory.seed = 42  
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def prepare_features(df):
    df['title_text'] = df['title'] + ' ' + df['text']

    df["text_word_count"] = df["text"].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
    df["title_word_count"] = df["title"].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
    
    df["text_unique_words"] = df["text"].apply(lambda x: len(set(x.lower().split())) if pd.notnull(x) else 0)
    
    df["text_char_count"] = df["text"].apply(lambda x: len(x) - x.count(' ') if pd.notnull(x) else 0)
    df["title_char_count"] = df["title"].apply(lambda x: len(x) - x.count(' ') if pd.notnull(x) else 0)

    # Applying the functions to the text column
    df['avg_word_length'] = df['text'].apply(lambda x: avg_word_length(str(x)))
    df['sentence_count'] = df['text'].apply(lambda x: sentence_count(str(x)))
    df['special_char_count'] = df['text'].apply(lambda x: special_char_count(str(x)))
    df['language'] = df['text'].astype(str).apply(detect_language)

    return df


# `text_cleaner` function
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
    no_stopWords_lemma_token = [token.lower() for token in lemma_token if token not in stopWords]
    # Remove punctuations
    clean_doc = [token for token in no_stopWords_lemma_token if token not in punctuations]
    # Use the `.join` method on `text` to convert string
    joined_clean_doc = " ".join(clean_doc)
    # Use `re.sub()` to substitute multiple spaces or dots`[\.\s]+` to single space `' '
    sub_doc = re.sub(r'[\.\s]+', ' ', joined_clean_doc)
    # Use `re.sub()` to substitute ■ to an empty string `' '
    final_doc = re.sub(r'[\■🚨]+', '', sub_doc)
    return final_doc


def apply_text_cleaner(df, column):
    # Progress bar
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Progress"):
        results.append(text_cleaner(row[column]))

    df['title_text_clean'] = results

    return df


def remove_outliers(features, target, num_cols, threshold=3, iterations=1):
    """
    Entfernt Ausreißer aus einem DataFrame anhand der Median Absolute Deviation (MAD)
    
    Parameter:
    df : pd.DataFrame - Der Eingabe-DataFrame.
    threshold : float - Der Schwellenwert für MAD (Standard: 3).
    iterations : int - Anzahl der Iterationen, um Ausreißer schrittweise zu entfernen (Standard: 1).
    
    Rückgabe:
    pd.DataFrame - DataFrame ohne Ausreißer.
    """
    features_sampled = features.copy()

    print((features_sampled.index != target.index).sum())

    # remove outliers with median absolute deviation
    for _ in range(iterations):
        median = features_sampled[num_cols].median()
        mad = (features_sampled[num_cols] - median).abs().median()

        mad_distance = (features_sampled[num_cols] - median).abs() / mad
        features_sampled = features_sampled[(mad_distance < threshold).all(axis=1)]

    # remove texts with less than 6 words
    features_sampled = features_sampled[features_sampled['text_word_count'] > 5]

    # remove non english texts
    features_sampled = features_sampled[features_sampled['language'] == 'en']


    target = target[features_sampled.index]
    
    return features_sampled, target


def evaluate(y_test, y_pred):
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')