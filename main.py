import socket
import threading
from tkinter import *
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
import random
import string
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageTk
import warnings
import pickle
import nltk
# nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
from tensorflow.python.framework import ops


class GUI:
    def __init__(self):
        self.Window = Tk()
        self.Window.withdraw()

        # login window
        self.login = Toplevel()

        # set the title
        self.login.title("Login")
        self.login.resizable(width=False, height=False)
        self.login.configure(width=400, height=300)

        # create a Label
        self.pls = Label(self.login, text="What should I call you?", justify=CENTER, font="Helvetica 14 bold")
        self.pls.place(relheight=0.15, relx=0.2, rely=0.07)

        # create a Label
        self.labelName = Label(self.login, text="Name: ", font="Helvetica 12")
        self.labelName.place(relheight=0.2, relx=0.1, rely=0.2)

        # create a entry box for  tyoing the message
        self.entryName = Entry(self.login, font="Helvetica 14")
        self.entryName.place(relwidth=0.4, relheight=0.12, relx=0.35, rely=0.2)

        # set the focus of the cursor
        self.entryName.focus()

        # create a Continue Button along with action
        self.go = Button(self.login, text="Next", font="Helvetica 14 bold",
                         command=lambda: self.goAhead(self.entryName.get()))
        self.go.place(relx=0.4, rely=0.55)
        self.Window.mainloop()

    def goAhead(self, name):
        self.login.destroy()
        self.layout(name)

    # The main layout of the chat
    def layout(self, name):
        # avatar
        # image = Image.open('face.png')
        # image.thumbnail((300, 300), Image.ANTIALIAS)
        # photo = ImageTk.PhotoImage(image)
        # label_image = tkinter.Label(image=photo)
        # label_image.grid(column=1, row=0)

        self.name = name

        # to show chat window
        self.Window.deiconify()
        self.Window.title("Varothex")
        self.Window.resizable(width=False, height=False)
        self.Window.configure(width=470, height=550, bg="#17202A")
        self.labelHead = Label(self.Window, bg="#17202A", fg="#EAECEE", text=self.name, font="Helvetica 13 bold",
                               pady=5)
        self.labelHead.place(relwidth=1)
        self.line = Label(self.Window, width=450, bg="#ABB2B9")
        self.line.place(relwidth=1, rely=0.07, relheight=0.012)
        self.textCons = Text(self.Window, width=20, height=2, bg="#17202A", fg="#EAECEE", font="Helvetica 14", padx=5,
                             pady=5)
        self.textCons.place(relheight=0.745, relwidth=1, rely=0.08)
        self.labelBottom = Label(self.Window, bg="#ABB2B9", height=80)
        self.labelBottom.place(relwidth=1, rely=0.825)
        self.entryMsg = Entry(self.labelBottom, bg="#2C3E50", fg="#EAECEE", font="Helvetica 13")

        # place the given widget into the gui window
        self.entryMsg.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.entryMsg.focus()

        # create a Send Button
        self.buttonMsg = Button(self.labelBottom, text="Send", font="Helvetica 10 bold", width=20, bg="#ABB2B9",
                                command=lambda: self.sendButton(self.entryMsg.get()))
        self.buttonMsg.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)
        self.textCons.config(cursor="arrow")

        # create a scroll bar
        scrollbar = Scrollbar(self.textCons)

        # place the scroll bar into the gui window
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.config(command=self.textCons.yview)
        self.textCons.config(state=DISABLED)

        self.greet()

    # function to basically start the thread for sending messages
    def sendButton(self, msg):
        self.textCons.config(state=DISABLED)
        self.msg = msg
        self.entryMsg.delete(0, END)
        snd = threading.Thread(target=self.sendMessage)
        snd.start()

    def greet(self):
        self.textCons.config(state=DISABLED)
        message = "Varothex: Hello there, my name is Varothex. I'm here for you! If you want to exit, " \
                  "type „bye”! "
        self.textCons.config(state=NORMAL)
        self.textCons.insert(END, message + "\n\n")
        self.textCons.config(state=DISABLED)
        self.textCons.see(END)

    # function to send messages
    def sendMessage(self):
        self.textCons.config(state=DISABLED)
        while True:
            message = f"{self.name}: {self.msg}"
            self.textCons.config(state=NORMAL)
            self.textCons.insert(END, message + "\n\n")
            self.textCons.config(state=DISABLED)
            self.textCons.see(END)
            self.send()
            break

    def send(self):
        self.textCons.config(state=DISABLED)
        message = "Varothex: " + response(self.msg)                                # afisam raspunsul
        self.textCons.config(state=NORMAL)
        self.textCons.insert(END, message + "\n\n")
        self.textCons.config(state=DISABLED)
        self.textCons.see(END)


warnings.filterwarnings('ignore')
stemmer = LancasterStemmer()

with open('intents.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?', "'m", "'re", "'s", ')', ',', '.', ':']  # eventual aici putem defini o lista de STOPWORDS

# pentru fiecare fraza din sabloanele corespunzatoare unei intentii
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenizare
        w = nltk.word_tokenize(pattern)
        # adaugam cuvintele la o lista globala cu toate cuvintele din texte
        words.extend(w)
        # adaugam textul tokenizat la multimea de documente
        documents.append((w, intent['tag']))
        # adaugam clasa (tipul de intentie) la lista de clase existente
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# aplicam stemming si lowercasing pentru fiecare cuvant intalnit, ignoram stopwords
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))  # eliminam duplciate

classes = sorted(list(set(classes)))

# construim datele de antrenare
training = []
output = []
# un vector de 0 de lungime egala cu numarul de clase (vom folosi reprezentarea one-hot a claselor)
output_empty = [0] * len(classes)

# bag of words pentru fiecare fraza
for doc in documents:
    # initializam bag of words
    bag = []
    # lista de tokeni pentru fraza curenta
    pattern_words = doc[0]
    # aplicam stemming
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # construim vectorul binar pentru bag of words
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output este '1' pentru tagul corespunzator frazei si '0' pentru celelalte (one-hot)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle + transformare in np.array
random.shuffle(training)
training = np.array(training)

# feature-urile de antrenare
train_x = list(training[:, 0])
# etichete/labels (ce dorim sa returneze modelul)
train_y = list(training[:, 1])

ops.reset_default_graph()  # resetam starea engine-ului TensorFlow

# definim reteaua
net = tflearn.input_data(shape=[None, len(train_x[0])])  # toti vectorii de bag-of-words au aceasta dimensiune
net = tflearn.fully_connected(net, 8)  # strat feed-forward ascuns cu 8 noduri
net = tflearn.fully_connected(net, 8)  # strat feed-forward ascuns cu 8 noduri
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')  # numarul de noduri output = numarul de clase
net = tflearn.regression(net)  # folosim acest strat de logistic regression pentru a extrage probabilitatile claselor

# definim modelul final
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# incepem antrenarea folosind coborarea pe gradient
# trecem prin model cate 8 fraze odata (batch size = 8)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

import pickle
pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open("training_data", "wb"))

data = pickle.load(open("training_data", "rb"))

words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

with open('intents.json') as json_data:
    intents = json.load(json_data)

# incarcam weight-urile salvate pentru cel mai bun model de clasificare
model.load('model.tflearn')


def clean_up_sentence(sentence):
    # tokenizare
    sentence_words = nltk.word_tokenize(sentence)
    # stemming
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# reprezentare binara bag-of-words
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return np.array(bag)


ERROR_THRESHOLD = 0.25


def classify(sentence):
    # probabilitatile prezise de model
    results = model.predict([bow(sentence, words)])[0]
    # renuntam la intentiile cu probabilitate mica
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    # sortam descrescator intentiile dupa probabilitate
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return perechi (intentie, probabilitate)
    return return_list


def response(sentence):
    results = classify(sentence)
    # daca avem macar o intentie valida, o procesam pe cea cu probabilitate maxima
    if results:
        while results:
            for i in intents['intents']:
                # cautam in dictionarul de intentii tagul returnat
                if i['tag'] == results[0][0]:
                    # returnam un raspuns aleator corespunzator intentiei
                    return random.choice(i['responses'])

            results.pop(0)
            # daca nu am putut da un raspuns pentru aceasta intentie, trecem la urmatoarea cu probabilitate maxima

    return "Sorry, I don't understand."  # nu a putut fi stabilita o intentie pentru
    # fraza introdusa


# structura de date pentru context
context = {}


# retinem pentru un user contextul curent, in functie de un ID specific
def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # daca avem macar o intentie valida, o procesam pe cea cu probabilitate maxima
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # cautam in dictionarul de intentii tagul returnat
                if i['tag'] == results[0][0]:
                    # daca intentia curenta asteapta un context pentru a fi valida, verificam ca contextul actual sa
                    # fie indeplinit
                    if ('context_filter' not in i
                            or (userID in context and 'context_filter' in i and i['context_filter'] == context[
                                userID])):
                        if show_details:
                            print('tag:', i['tag'])
                        # daca intentia curenta actualizeaza contextul
                        if 'context_set' in i:
                            if show_details:
                                print('context:', i['context_set'])
                            context[userID] = i['context_set']
                        # returnam un raspuns aleator corespunzator intentiei
                        return random.choice(i['responses'])
            results.pop(0)
            # daca nu am putut da un raspuns pentru aceasta intentie, trecem la urmatoarea cu probabilitate maxima
    # else:
        # folosim vechiul cod

    return "Sorry, I wasn't trained about this subject."  # nu a putut fi stabilita o intentie pentru fraza introdusa


if __name__ == "__main__":
    g = GUI()
