"""
Load raw dataset, remove duplicates, perform train-, test-, val-split, preprocess data
and upload feature sets to s3.
"""

import os
import sys
import logging
from datetime import datetime

import yaml
from sklearn.model_selection import train_test_split

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))

from utils import preprocessing as prep
from utils import utility_functions as uf

config = uf.load_config()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

s3_bucket = config['mlflow']['model_bucket']
dataset_path = os.getenv('DATASET_PATH', 'datasets/WELFake_Dataset.csv')  # raw dataset


# Load raw dataset
# print('load raw dataset')
df = uf.load_file_s3(s3_bucket, dataset_path, 'csv')
logging.info('raw dataset was loaded')


# Remove duplicates so that they do not end up in the validation and test data set

logging.info('remove duplicates')
df = df.drop_duplicates(subset='text').reset_index().drop(columns='index')
df = df.drop_duplicates(subset='title').reset_index().drop(columns='index')
logging.info('duplicates removed')

df = df[:100]

# Perform train-, val- and test-split

logging.info('perform tain-, val-, test-split')
X = df.drop(columns='label')
y = df.loc[:, 'label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=42,
)

X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

logging.info('tain-, val-, test-split done')
logging.info("Shapes after Split:")
logging.info('train data:', X_train.shape, y_train.shape)
logging.info('val data:', X_val.shape, y_val.shape)
logging.info('test data:', X_test.shape, y_test.shape)

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

logging.info('clean data')
X_train, y_train = prep.clean_data(X_train, y_train)
X_val, y_val = prep.clean_data(X_val, y_val)
X_test, y_test = prep.clean_data(X_test, y_test)
logging.info('data cleaned')

logging.info('prepare data')
X_train = prep.prepare_features(X_train)
X_val = prep.prepare_features(X_val)
X_test = prep.prepare_features(X_test)
logging.info('data prepared')


# Apply text cleaner on column 'title_text': Using NLP-Steps include Lemmatization, removing stop words, removing punctuations and substitute multiple spaces or dots to single space. Returns new column 'title_text_clean' to DataFrame with the cleaned text.

logging.info('apply text cleaner')
X_train = prep.apply_text_cleaner(X_train, column='title_text')
X_val = prep.apply_text_cleaner(X_val, column='title_text')
X_test = prep.apply_text_cleaner(X_test, column='title_text')
logging.info('text cleaner applied')

# Upload features and target to s3

logging.info('upload features to s3')
uf.upload_to_s3(X_train, s3_bucket, 'X_train')
uf.upload_to_s3(X_val, s3_bucket, 'X_val')
uf.upload_to_s3(X_test, s3_bucket, 'X_test')

logging.info('upload target to s3')
uf.upload_to_s3(y_train, s3_bucket, 'y_train')
uf.upload_to_s3(y_val, s3_bucket, 'y_val')
uf.upload_to_s3(y_test, s3_bucket, 'y_test')

date = datetime.now().strftime("%Y-%m-%d")

config['data']['X_train'] = f'datasets/X_train_{date}.parquet'
config['data']['X_val'] = f'datasets/X_val_{date}.parquet'
config['data']['X_test'] = f'datasets/X_test_{date}.parquet'

config['data']['y_train'] = f'datasets/y_train_{date}.csv'
config['data']['y_val'] = f'datasets/y_val_{date}.csv'
config['data']['y_test'] = f'datasets/y_test_{date}.csv'

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'config', 'vars.yaml')

with open(config_path, "w", encoding='utf-8') as file:
    yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
