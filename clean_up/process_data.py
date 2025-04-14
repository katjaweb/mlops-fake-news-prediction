# Imports

import os
import pandas as pd
import spacy
import nltk
import string
import time
import re
import boto3
import io
from datetime import datetime

from tqdm import tqdm
from nltk.corpus import stopwords
from textblob import TextBlob
from langdetect import detect, DetectorFactory

from sklearn.model_selection import train_test_split

s3_bucket = os.getenv('S3_BUCKET', 'fake-news-prediction')
dataset_path = os.getenv("DATASET_PATH", 'datasets/WELFake_Dataset.csv') # raw dataset

nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
stopWords = stopwords.words('english')
punctuations = string.punctuation

# Create functions

# Function for data cleaning
def clean_data(features, target):
    # drop unused column
    features = features.drop(columns='Unnamed: 0')
    # Reverse labels into: fake=1, real=0
    target = 1 - target
    # Fill Nan-values with an empty string
    features[['title', 'text']] = features[['title', 'text']].fillna("")

    return features, target

# Function for calculating sentiment scores
def get_sentiment(text):
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity

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
    # df['language'] = df['text'].astype(str).apply(detect_language)
    df['sentiment'] = df['text'].apply(get_sentiment)

    df = df.drop(['title', 'text'], axis=1)

    return df


# `text_cleaner` function
def text_cleaner(text):
    """Clean the text using NLP-Steps.

    Steps include: Lemmatization, removing stop words, removing punctuations 

    Args:
        sentence (str): The uncleaned text.

    Returns:
        str: The cleaned text.

    """

    # Create the Doc object named `text` from `sentence` using `nlp()`
    doc = nlp(text)
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

    df = df.drop(['title_text'], axis=1)

    return df

def load_file_s3(s3_bucket, dataset_path, type):
    s3 = boto3.client('s3')

    if type == 'csv':
        buffer = io.BytesIO()
        s3.download_fileobj(s3_bucket, dataset_path, buffer)
        buffer.seek(0)
        data = pd.read_csv(buffer)
        return data
    else:
        buffer = io.BytesIO()
        s3.download_fileobj(s3_bucket, dataset_path, buffer)
        buffer.seek(0)
        data = pd.read_parquet(buffer)
        return data

def upload_to_s3(file, s3_bucket, dataset):
    s3_client = boto3.client('s3')

    # actual date
    date = datetime.now().strftime("%Y-%m-%d")

    if isinstance(file, pd.Series):
        buffer = io.BytesIO()
        file.to_csv(buffer, index=False)
        buffer.seek(0)
        # create file name with date
        file_key = f"datasets/{dataset}_{date}.csv"
        print(file_key)
    else:
        buffer = io.BytesIO()
        file.to_parquet(buffer, index=False)
        buffer.seek(0)
        # create file name with date
        file_key = f"datasets/{dataset}_{date}.parquet"
        print(file_key)

    s3_client.put_object(Bucket=s3_bucket, Key=file_key, Body=buffer.getvalue())
    print(f"File saved under {file_key} in {s3_bucket}.")


# Load raw dataset

print('load raw dataset')
df = load_file_s3(s3_bucket, dataset_path, 'csv')
print('raw dataset was loaded')


# Remove duplicates so that they do not end up in the validation and test data set

print('remove duplicates')
df = df.drop_duplicates(subset='text').reset_index().drop(columns='index')
df = df.drop_duplicates(subset='title').reset_index().drop(columns='index')
print('duplicates removed')

# df = df[:1000]


# Perform train-, val- and test-split

print('perform tain-, val-, test-split')
X = df.drop(columns='label')
y = df.loc[:, 'label']

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                    y_train, 
                                                    test_size=0.25, 
                                                    random_state=42)

print('tain-, val-, test-split done')

# clean data: drop unused column 'Unnamed: 0', reverse labels into: fake=1 and real=0, fill Nan-values with an empty string
# 
# prepare features: Create new features
# - title_text: concatenate columns title and text
# - text_word_count: # of words in text column
# - title_word_count: # of words in title column
# - text_unique_words: # of uniqe words in text column
# - text_char_count: # of characters in text column
# - title_char_count: # of characters in title column
# - avg_word_length: average word length in text column
# - sentence_count: # of sentences in text column
# - special_char_count: # of special characters in text column
# - language: estimated language for text column
# - sentiment: calculate sentiment scores

print('clean data')
X_train, y_train = clean_data(X_train, y_train)
X_val, y_val = clean_data(X_val, y_val)
X_test, y_test = clean_data(X_test, y_test)
print('data cleaned')

print('prepare data')
X_train = prepare_features(X_train)
X_val = prepare_features(X_val)
X_test = prepare_features(X_test)
print('data prepared')


# Apply text cleaner on column 'title_text': Using NLP-Steps include Lemmatization, removing stop words, removing punctuations and substitute multiple spaces or dots to single space. Returns new column 'title_text_clean' to DataFrame with the cleaned text.

print('apply text cleaner')
X_train = apply_text_cleaner(X_train, column='title_text')
X_val = apply_text_cleaner(X_val, column='title_text')
X_test = apply_text_cleaner(X_test, column='title_text')
print('text cleaner applied')

# Upload features and target to s3

print('upload features to s3')
upload_to_s3(X_train, s3_bucket, "X_train")
upload_to_s3(X_val, s3_bucket, "X_val")
upload_to_s3(X_test, s3_bucket, "X_test")

print('upload target to s3')
upload_to_s3(y_train, s3_bucket, "y_train")
upload_to_s3(y_val, s3_bucket, "y_val")
upload_to_s3(y_test, s3_bucket, "y_test")
