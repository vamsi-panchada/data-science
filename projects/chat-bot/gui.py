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
    res = model.predict(np.array([predicted]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list=[]
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list
    

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
    
import tkinter
from tkinter import *

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "You: " + msg + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12 ))
    
        ints = predict_class(msg)
        res = get_response(ints, intents)
        
        ChatBox.insert(END, "Bot: " + res + '\n\n')
            
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)
 

root = Tk()
root.title("Chatbot")
root.geometry("400x500")
root.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatBox.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

#Create the box to enter message
EntryBox = Text(root, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Create Button to send message
SendButton = Button(root, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#f9a602", activebackground="#3c9d9b",fg='#000000',
                    command= send )


#Place all components on the screen
scrollbar.place(x=376,y=6, height=426)
ChatBox.place(x=6,y=6, height=426, width=370)
EntryBox.place(x=6, y=441, height=50, width=320)
SendButton.place(x=335, y=441, height=50, width=60)

root.mainloop()

