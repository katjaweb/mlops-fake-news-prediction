"""
Runs several tests for preprocessing
"""

# pylint: disable=unsubscriptable-object
import os
import sys
import pandas as pd

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from utils import preprocessing as prep


def test_avg_word_length():
    """
    tests whether the average word length has been calculated correctly
    """

    text = ' When Donald Trump explained why he had put a 90-day pause on his sweeping tariff scheme, he told reporters that it was because countries had started to get yippy. But what exactly did the president mean by this; was the keen golfer referring to the so-called yips that so often hinder a clean putt? Or something else entirely? Dr Robbie Love, a senior lecturer in English Language and Linguistics at Aston University, tells me that "yip" can be traced back as far as the 1400s, where there is evidence of it being used to refer to the cheeping sound of a newly hatched bird. But, by the 20th Century this had been adapted to refer to "high-pitched barking of small dogs" and also "any kind of short, high-pitched cry, including those made by people", Love explains. "The adjective ‘yippy’ seems to be derived from this sense - describing [in this case in humans] a quality of excitable/anxious [and also perhaps annoying] yelping, like a dog." The expert, though, stresses that he cant be sure whether this is exactly what Trump intended to imply when he addressed reporters outside the White House. He may have meant that hed been inundated with communications about the tariffs, and/or perhaps he meant something closer to skittish, Love observes. Or perhaps it reveals something about how he regards the leaders of other countries.'

    expected_word_length = 4.855263157894737

    actual_word_length = prep.avg_word_length(text)

    assert expected_word_length == actual_word_length


def test_prepare_features():
    """
    tests whether the new features have been created.
    """
    dummy_news = {
        "title": 'Yippy - what exactly was Donald Trump talking about?',
        "text": ' When Donald Trump explained why he had put a 90-day pause on his sweeping tariff scheme, he told reporters that it was because countries had started to get yippy. But what exactly did the president mean by this; was the keen golfer referring to the so-called yips that so often hinder a clean putt? Or something else entirely? Dr Robbie Love, a senior lecturer in English Language and Linguistics at Aston University, tells me that "yip" can be traced back as far as the 1400s, where there is evidence of it being used to refer to the cheeping sound of a newly hatched bird. But, by the 20th Century this had been adapted to refer to "high-pitched barking of small dogs" and also "any kind of short, high-pitched cry, including those made by people", Love explains. "The adjective ‘yippy’ seems to be derived from this sense - describing [in this case in humans] a quality of excitable/anxious [and also perhaps annoying] yelping, like a dog." The expert, though, stresses that he cant be sure whether this is exactly what Trump intended to imply when he addressed reporters outside the White House. He may have meant that hed been inundated with communications about the tariffs, and/or perhaps he meant something closer to skittish, Love observes. Or perhaps it reveals something about how he regards the leaders of other countries.',
    }

    input_df = pd.DataFrame(dummy_news, index=[0])

    result_df = prep.prepare_features(pd.DataFrame(input_df))

    print(result_df)

    assert isinstance(result_df, pd.DataFrame)
    assert 'title_text' in result_df.columns
    assert 'text_word_count' in result_df.columns
    assert 'title_word_count' in result_df.columns
    assert 'text_unique_words' in result_df.columns
    assert 'text_char_count' in result_df.columns
    assert 'title_char_count' in result_df.columns
    assert 'avg_word_length' in result_df.columns
    assert 'sentence_count' in result_df.columns
    assert 'special_char_count' in result_df.columns
    assert 'sentiment' in result_df.columns
    assert isinstance(result_df['title_text'].iloc[0], str)
    assert result_df.shape[1] == 10


def test_apply_text_cleaner():
    """
    tests whether the text has been cleaned up using NLP steps.
    """
    dummy_news = {
        "title_text": ' When Donald Trump explained why he had put a 90-day pause on his sweeping tariff scheme, he told reporters that it was because countries had started to get yippy. But what exactly did the president mean by this; was the keen golfer referring to the so-called yips that so often hinder a clean putt? Or something else entirely? Dr Robbie Love, a senior lecturer in English Language and Linguistics at Aston University, tells me that "yip" can be traced back as far as the 1400s, where there is evidence of it being used to refer to the cheeping sound of a newly hatched bird. But, by the 20th Century this had been adapted to refer to "high-pitched barking of small dogs" and also "any kind of short, high-pitched cry, including those made by people", Love explains. "The adjective ‘yippy’ seems to be derived from this sense - describing [in this case in humans] a quality of excitable/anxious [and also perhaps annoying] yelping, like a dog." The expert, though, stresses that he cant be sure whether this is exactly what Trump intended to imply when he addressed reporters outside the White House. He may have meant that hed been inundated with communications about the tariffs, and/or perhaps he meant something closer to skittish, Love observes. Or perhaps it reveals something about how he regards the leaders of other countries.',
    }

    df_test = pd.DataFrame(dummy_news, index=[0])

    df_test = prep.apply_text_cleaner(df_test, 'title_text')

    actual_result = df_test['title_text_clean'].iloc[0]

    expected_result = ' donald trump explain put 90 day pause sweeping tariff scheme tell reporter country start get yippy exactly president mean keen golfer refer call yip often hinder clean putt else entirely dr robbie love senior lecturer english language linguistics aston university tell yip trace back far 1400 evidence use refer cheep sound newly hatch bird 20th century adapt refer high pitch barking small dog also kind short high pitch cry include make people love explain adjective yippy seem derive sense describe case human quality excitable anxious also perhaps annoying yelping like dog expert though stress sure whether exactly trump intend imply address reporter outside white house may mean inundate communication tariff and/or perhaps mean close skittish love observe perhaps reveal regard leader country'

    assert actual_result == expected_result


def test_sentence_count():
    """
    tests whether the number of sentences has been calculated correctly.
    """
    text = ' When Donald Trump explained why he had put a 90-day pause on his sweeping tariff scheme, he told reporters that it was because countries had started to get yippy. But what exactly did the president mean by this; was the keen golfer referring to the so-called yips that so often hinder a clean putt? Or something else entirely? Dr Robbie Love, a senior lecturer in English Language and Linguistics at Aston University, tells me that "yip" can be traced back as far as the 1400s, where there is evidence of it being used to refer to the cheeping sound of a newly hatched bird. But, by the 20th Century this had been adapted to refer to "high-pitched barking of small dogs" and also "any kind of short, high-pitched cry, including those made by people", Love explains. "The adjective ‘yippy’ seems to be derived from this sense - describing [in this case in humans] a quality of excitable/anxious [and also perhaps annoying] yelping, like a dog." The expert, though, stresses that he cant be sure whether this is exactly what Trump intended to imply when he addressed reporters outside the White House. He may have meant that hed been inundated with communications about the tariffs, and/or perhaps he meant something closer to skittish, Love observes. Or perhaps it reveals something about how he regards the leaders of other countries.'

    actual_result = prep.sentence_count(text)

    expected_result = 9

    assert actual_result == expected_result


def test_special_character_count():
    """
    tests whether the number of special characters has been calculated correctly.
    """
    text = 'DHL Express, a division of Germany’s Deutsche Post, said it would suspend global business-to-consumer shipments worth over $800 to individuals in the United States from April 21, as US customs regulatory changes have lengthened clearance. The notice on the company website was not dated, but its metadata showed it was compiled on Saturday. DHL blamed the halt on new US customs rules which require formal entry processing on all shipments worth over $800. The minimum had been $2,500 until a change on April 5. DHL said business-to-business shipments would not be suspended but could face delays. Shipments under $800 to either businesses or consumers were not affected by the changes. The move is a temporary measure, the company said in its statement.'

    actual_result = prep.special_char_count(text)

    expected_result = 22

    assert actual_result == expected_result
