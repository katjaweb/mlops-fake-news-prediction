# Fake News detection Web-service

This project addresses the growing challenge of online disinformation by providing a web-based tool to detect fake news. Built with Flask, the service uses a supervised machine learning model (LightGBM) and natural language processing techniques to classify English news articles as real or fake. It also provides a probability score to indicate prediction confidence.

# Key features:
* Text preprocessing with tokenization and stopword removal
* Feature extraction using Bag-of-Words (BoW)
* Model trained on a labeled dataset of English news

The goal is to support journalists, researchers, and the public in critically assessing the credibility of online information.

About Dataset

(WELFake) is a dataset of 72,134 news articles with 35,028 real and 37,106 fake news. For this, authors merged four popular news datasets (i.e. Kaggle, McIntire, Reuters, BuzzFeed Political) to prevent over-fitting of classifiers and to provide more text data for better ML training.

Dataset contains four columns: Serial number (starting from 0); Title (about the text news heading); Text (about the news content); and Label (0 = fake and 1 = real).

There are 78098 data entries in csv file out of which only 72134 entries are accessed as per the data frame.

Published in:
IEEE Transactions on Computational Social Systems: pp. 1-13 (doi: 10.1109/TCSS.2021.3068519).
