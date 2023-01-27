import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import ssl

from keras.models import load_model
import json
import random

lemmatizer = WordNetLemmatizer()
model = load_model('Data/chatbot_model.h5')
intentFile = open('Data/intents.json').read()
intents = json.loads(intentFile)
words = pickle.load(open('Data/words.pkl', 'rb'))
classes = pickle.load(open('Data/classes.pkl', 'rb'))

def clean_sentence(sentence):
    try:
        sentence_words = nltk.word_tokenize(sentence)
        try:
            sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        except:
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context

            nltk.download('wordnet')
            nltk.download('omw-1.4')
            sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    except:
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download('punkt')
        sentence_words = nltk.word_tokenize(sentence)
        try:
            sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        except:
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context

            nltk.download('wordnet')
            nltk.download('omw-1.4')
            sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
            
    return sentence_words


def bag_of_words(sentence, words, show_details=True):
    sentence_words = clean_sentence(sentence)
    bag = [0]*len(words)
    for sentence_word in sentence_words:
        for i, word in enumerate(words):
            if word == sentence_word:
                bag[i]=1
                if show_details:
                    print('found in bag : '+word)
    return np.array(bag)


def predict_class(sentence):
    predicted = bag_of_words(sentence, words, show_details=False)
    
