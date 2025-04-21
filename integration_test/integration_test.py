"""
integration test for web-service and docker
"""
import requests
from deepdiff import DeepDiff

# print('Starting test')
news = {
    "title": 'Yippy - what exactly was Donald Trump talking about?',
    "text": ' When Donald Trump explained why he had put a 90-day pause on his sweeping tariff scheme, he told reporters that it was because countries had started to get yippy. But what exactly did the president mean by this; was the keen golfer referring to the so-called yips that so often hinder a clean putt? Or something else entirely? Dr Robbie Love, a senior lecturer in English Language and Linguistics at Aston University, tells me that "yip" can be traced back as far as the 1400s, where there is evidence of it being used to refer to the cheeping sound of a newly hatched bird. But, by the 20th Century this had been adapted to refer to "high-pitched barking of small dogs" and also "any kind of short, high-pitched cry, including those made by people", Love explains. "The adjective ‘yippy’ seems to be derived from this sense - describing [in this case in humans] a quality of excitable/anxious [and also perhaps annoying] yelping, like a dog." The expert, though, stresses that he cant be sure whether this is exactly what Trump intended to imply when he addressed reporters outside the White House. He may have meant that hed been inundated with communications about the tariffs, and/or perhaps he meant something closer to skittish, Love observes. Or perhaps it reveals something about how he regards the leaders of other countries.',
}

# print('print news:', news)

URL = 'http://localhost:9696/predict'
actual_response = requests.post(URL, json=news, timeout=10).json()

expected_response = {"label":"real news","model_version":"2b20e3e8444e4d3b8023a4a6c0112117","probability being fake":0.059,"probability being real":0.941}

# print('Status code:', actual_response.status_code)
print('actual response:', actual_response)
print('expected response:', expected_response)

diff = DeepDiff(actual_response, expected_response, significant_digits=1)
print(f"diff={diff}")

assert "type_changes" not in diff
assert "values_changed" not in diff
assert actual_response == expected_response

print('all good.')
