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

words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
#print(len(documents), len(words), len(classes))

pickle.dump(words, open('Data/words.pkl', 'wb'))
pickle.dump(classes, open('Data/classes.pkl', 'wb'))

training = []
output_empty = [0]*len(classes)

for document in documents:
    bag = []
    pattern_words = document[0]
    try:
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    except:
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download('wordnet')
        nltk.download('omw-1.4')
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    
    training.append([bag, output_row])

print(training)
