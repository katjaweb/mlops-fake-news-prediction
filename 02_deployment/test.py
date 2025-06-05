"""
Sends a test POST request to the local prediction API endpoint with sample news data.

The script constructs a fake news article payload, sends it to the '/predict' route
on a locally running server, and prints out the HTTP status code, raw response text,
and parsed JSON response containing the prediction results.
"""

# pylint: disable=line-too-long

import requests

# print('Starting test')
news = {
    "title": 'Yippy - what exactly was Donald Trump talking about?',
    "text": ' When Donald Trump explained why he had put a 90-day pause on his sweeping tariff scheme, he told reporters that it was because countries had started to get yippy. But what exactly did the president mean by this; was the keen golfer referring to the so-called yips that so often hinder a clean putt? Or something else entirely? Dr Robbie Love, a senior lecturer in English Language and Linguistics at Aston University, tells me that "yip" can be traced back as far as the 1400s, where there is evidence of it being used to refer to the cheeping sound of a newly hatched bird. But, by the 20th Century this had been adapted to refer to "high-pitched barking of small dogs" and also "any kind of short, high-pitched cry, including those made by people", Love explains. "The adjective ‘yippy’ seems to be derived from this sense - describing [in this case in humans] a quality of excitable/anxious [and also perhaps annoying] yelping, like a dog." The expert, though, stresses that he cant be sure whether this is exactly what Trump intended to imply when he addressed reporters outside the White House. He may have meant that hed been inundated with communications about the tariffs, and/or perhaps he meant something closer to skittish, Love observes. Or perhaps it reveals something about how he regards the leaders of other countries.',
}

URL = 'http://localhost:9696/predict'  # to test locally
# HOST = 'fake-news-service-env.eba-cryzmisk.eu-west-1.elasticbeanstalk.com'
# URL = f'http://{HOST}/predict'
response = requests.post(URL, json=news, timeout=10)

print('Status code:', response.status_code)
print('Text response:', response.json())
