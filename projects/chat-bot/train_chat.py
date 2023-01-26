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

#print(training)

#for train in training:
#    print(train)
#    print('--'*25)

random.shuffle(training)
training = np.array(training, dtype=list)

train_x = list(training[:, 0])
train_y = list(training[:, 1])
#print(train_x)
#print('++==++'*25)
#print(train_y)
print('Training Data created succesful')

#model creation
#no.of layers = 3
#no.of nuerons in layer1 = 128
#no.of nuerons in layer2 = 64
#no.of nuerons in layer3 = no.of intents in predict output.

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

trainded_model = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('Data/chatbot_model.h5', trainded_model)
print('Training Completed')
