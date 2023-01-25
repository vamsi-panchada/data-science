import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
import nltk
import ssl
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

words = []
classes = []
documents = []

intentFile = open('Data/intents.json').read()
intents = json.loads(intentFile)
#print(intents)

ingore_chars = ['!', '?', ',', '.']
for intent in intents['intents']:
#    print(intent)
#    print('--'*25)
    for pattern in intent['patterns']:
        try:
            word = nltk.word_tokenize(pattern)
        except:
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context

            nltk.download('punkt')
        words.extend(word)
        documents.append((word, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#print(words, documents, classes, sep ='--'*50)
try:
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ingore_chars]
except:
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('wordnet')
    nltk.download('omw-1.4')
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ingore_chars]
        
#print(words)

