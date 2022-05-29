import socket
import threading
from tkinter import *
from PIL import Image, ImageTk
import warnings
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import random
import json
from tensorflow.python.framework import ops
import pickle


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
        self.pls = Label(self.login, text="What should I call you?", justify=CENTER, font="Roboto 14 bold")
        self.pls.place(relheight=0.15, relx=0.2, rely=0.07)

        # create a Label
        self.labelName = Label(self.login, text="Name: ", font="Roboto 12")
        self.labelName.place(relheight=0.2, relx=0.1, rely=0.2)

        # create a entry box for tyoing the message
        self.entryName = Entry(self.login, font="Roboto 14")
        self.entryName.place(relwidth=0.4, relheight=0.12, relx=0.35, rely=0.2)

        # set the focus of the cursor
        self.entryName.focus()

        # create a Continue Button along with action
        self.go = Button(self.login, text="Next", font="Roboto 14 bold",
                         command=lambda: self.goAhead(self.entryName.get()))
        self.login.bind('<Return>', lambda event: self.goAhead(self.entryName.get()))
        self.go.place(relx=0.4, rely=0.55)
        self.Window.mainloop()

    def goAhead(self, name):
        self.login.destroy()
        self.layout(name)

    # The main layout of the chat
    def layout(self, name):
        self.name = name

        self.Window.deiconify()
        self.Window.title("Varothex")
        self.Window.resizable(width=False, height=False)
        self.Window.configure(width=1200, height=700, bg="#17202A")

        self.labelHead = Label(self.Window, bg="#17202A", fg="#EAECEE", text="avatar", font="Roboto 13 bold", pady=5)
        self.labelHead.place(relwidth=1, relheight=0.19)

        # TODO avatar

        self.line = Label(self.Window, width=450, bg="#ABB2B9")
        self.line.place(relwidth=1, rely=0.19, relheight=0.012)

        # chat window
        self.chatWindow = Text(self.Window, width=20, height=2, bg="#17202A", fg="#EAECEE", font="Roboto 14", padx=5,
                             pady=5)
        self.chatWindow.place(relheight=0.625, relwidth=0.985, rely=0.2)

        self.labelBottom = Label(self.Window, bg="#ABB2B9", height=80)
        self.labelBottom.place(relwidth=1, rely=0.825)

        # textbox
        self.textbox = Entry(self.labelBottom, bg="#2C3E50", fg="#EAECEE", font="Roboto 13")
        self.textbox.place(relwidth=0.74, relheight=0.06, relx=0.011, rely=0.008)
        self.textbox.focus()

        # send button
        self.buttonMsg = Button(self.labelBottom, text="Send", font="Roboto 14 bold", width=20, bg="#ABB2B9",
                                command=lambda: self.sendButton(self.textbox.get()))
        self.Window.bind('<Return>', lambda event: self.sendButton(self.textbox.get()))
        self.buttonMsg.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)
        self.chatWindow.config(cursor="arrow")

        # scroll bar
        scrollbar = Scrollbar(self.Window)
        scrollbar.place(relheight=0.625, relx=0.985, rely=0.2)
        scrollbar.config(command=self.chatWindow.yview)
        self.chatWindow.config(state=DISABLED)

        self.greet()

    # function to basically start the thread for sending messages
    def sendButton(self, msg):
        self.chatWindow.config(state=DISABLED)
        self.msg = msg
        self.textbox.delete(0, END)
        snd = threading.Thread(target=self.sendMessage)
        snd.start()

    def greet(self):
        self.chatWindow.config(state=DISABLED)
        message = "Varothex: Hello there, my name is Varothex. I'm here for you!"
        self.chatWindow.config(state=NORMAL)
        self.chatWindow.insert(END, message + "\n\n")
        self.chatWindow.config(state=DISABLED)
        self.chatWindow.see(END)

    # function to send messages
    def sendMessage(self):
        self.chatWindow.config(state=DISABLED)
        while True:
            message = f"{self.name}: {self.msg}"
            self.chatWindow.config(state=NORMAL)
            self.chatWindow.insert(END, message + "\n\n")
            self.chatWindow.config(state=DISABLED)
            self.chatWindow.see(END)
            self.botAnswer()
            break

    # return the bot answer
    def botAnswer(self):
        self.chatWindow.config(state=DISABLED)
        message = "Varothex: " + response(self.msg)
        self.chatWindow.config(state=NORMAL)
        self.chatWindow.insert(END, message + "\n\n")
        self.chatWindow.config(state=DISABLED)
        self.chatWindow.see(END)


warnings.filterwarnings('ignore')
stemmer = LancasterStemmer()

with open('intents.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?', "'m", "'re", "'s", ')', ',', '.', ':']  # STOPWORDS

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
# net = tflearn.fully_connected(net, 8)  # strat feed-forward ascuns cu 8 noduri
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')  # numarul de noduri output = numarul de clase
net = tflearn.regression(net)  # folosim acest strat de logistic regression pentru a extrage probabilitatile claselor

# definim modelul final
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# incepem antrenarea folosind coborarea pe gradient, trecem prin model cate 8 fraze odata (batch size = 8)
model.fit(train_x, train_y, n_epoch=1200, batch_size=8, show_metric=True)
model.save('model.tflearn')

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
    # TODO empty messages
    # if sentence == "":
    #     return "Did you say something?"

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
    # else:
        # folosim vechiul cod

    return "Sorry, I don't understand."  # nu a putut fi stabilita o intentie pentru fraza introdusa


if __name__ == "__main__":
    g = GUI()
