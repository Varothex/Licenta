import threading
from tkinter import *
import warnings
import nltk
from django.utils.datetime_safe import datetime
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import random
import json
from tensorflow.python.framework import ops
import pickle
from gtts import gTTS
import os
from pygame import mixer
import requests
import bs4
from time import *


# import PyAudio
# import speech_recognition as sr


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
        if name == "":
            name = "GUEST"
        self.login.destroy()
        self.layout(name)

    # The main layout of the chat
    def layout(self, name):
        self.name = name

        self.Window.deiconify()
        self.Window.title("Varothex")
        self.Window.resizable(width=False, height=False)
        self.Window.configure(width=1200, height=700, bg="#17202A")

        width = 0.7

        # avatar
        self.avatar = PhotoImage(file='avatar.png')  # need a reference to the image or it gets garbage collected
        self.avatarFrame = Label(self.Window, image=self.avatar)
        self.avatarFrame.place(relx=width, rely=0)

        # chat window
        self.chatWindow = Text(self.Window, width=20, height=2, bg="#17202A", fg="#EAECEE", font="Roboto 14", padx=5,
                               pady=5)
        self.chatWindow.place(relheight=0.825, relwidth=width)

        # scroll bar
        scrollbar = Scrollbar(self.Window)
        scrollbar.place(relheight=0.825, relx=width)
        scrollbar.config(command=self.chatWindow.yview)
        self.chatWindow.config(state=DISABLED)

        # bottom
        self.labelBottom = Label(self.Window, bg="#ABB2B9", height=80)
        self.labelBottom.place(relwidth=1, rely=0.825)

        # textbox
        self.textbox = Entry(self.labelBottom, bg="#2C3E50", fg="#EAECEE", font="Roboto 13")
        self.textbox.place(relwidth=0.68, relheight=0.06, relx=0.01, rely=0.015)
        self.textbox.focus()

        # send button
        self.buttonMsg = Button(self.labelBottom, text="Send", font="Roboto 14 bold", width=20, bg="#ABB2B9",
                                command=lambda: self.sendMessage(self.textbox.get()))
        self.Window.bind('<Return>', lambda event: self.sendMessage(self.textbox.get()))
        self.buttonMsg.place(relx=0.745, rely=0.015, relheight=0.06, relwidth=0.22)
        self.chatWindow.config(cursor="arrow")

        self.greet()

    def greet(self):
        self.chatWindow.config(state=DISABLED)
        time = strftime('%H:%M')
        message = time + " Varothex: Hello there, my name is Varothex. I'm here for you!"
        # myobj = gTTS(text="Hello there, my name is Varothex. I'm here for you!", lang='en', slow=False)
        # myobj.save("welcome.mp3")
        welcome = 'welcome.mp3'
        mixer.init()
        mixer.music.load(welcome)
        mixer.music.play()
        self.chatWindow.config(state=NORMAL)
        self.chatWindow.insert(END, message + "\n\n")
        self.chatWindow.config(state=DISABLED)
        self.chatWindow.see(END)

    # function to submit the messages
    def sendMessage(self, msg):
        self.chatWindow.config(state=DISABLED)
        self.msg = msg
        self.textbox.delete(0, END)
        snd = threading.Thread(target=self.showMessage)
        snd.start()

    # function to print the messages
    def showMessage(self):
        self.chatWindow.config(state=DISABLED)
        while True:
            time = strftime('%H:%M')
            message = time + f" {self.name}: {self.msg}"
            self.chatWindow.config(state=NORMAL)
            self.chatWindow.insert(END, message + "\n\n")
            self.chatWindow.config(state=DISABLED)
            self.chatWindow.see(END)
            self.botAnswer()
            break

    # function to return the bot answer
    def botAnswer(self):
        self.chatWindow.config(state=DISABLED)
        time = strftime('%H:%M')
        message, feeling = response(self.msg)
        message = time + " Varothex: " + message

        self.avatarFrame.configure(image=self.avatar)
        self.avatarFrame.image = self.avatar

        self.chatWindow.config(state=NORMAL)
        self.chatWindow.insert(END, message + "\n\n")
        self.chatWindow.config(state=DISABLED)
        self.chatWindow.see(END)

        # animations
        if feeling == "happy":
            self.avatar = PhotoImage(file='avatar_happy.png')
            self.avatarFrame.configure(image=self.avatar)
        elif feeling == "sad":
            self.avatar = PhotoImage(file='avatar_sad.png')
            self.avatarFrame.configure(image=self.avatar)
        else:
            self.avatar = PhotoImage(file='avatar.png')
            self.avatarFrame.configure(image=self.avatar)
        self.avatarFrame.image = self.avatar


# TODO mic input
# def get_audio():
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         audio = r.listen(source)
#         said = ""
#         try:
#             said = r.recognize_google(audio)
#             print(said)
#         except Exception as e:
#             print("Exception: " + str(e))
#     return said


warnings.filterwarnings('ignore')
stemmer = LancasterStemmer()

with open('intents.json') as json_data:
    intents = json.load(json_data)

# tokenizing words
words = []
classes = []
documents = []
ignore_words = ['?', "'", ',', '.', '!']  # STOPWORDS

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)  # adaugam cuvintele la o lista globala cu toate cuvintele din texte
        documents.append((w, intent['tag']))  # adaugam textul tokenizat la multimea de documente
        if intent['tag'] not in classes:  # adaugam clasa (tipul de intentie) la lista de clase existente
            classes.append(intent['tag'])

# aplicam stemming si lowercasing pentru fiecare cuvant intalnit, ignoram stopwords
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))  # eliminam duplciate
classes = sorted(list(set(classes)))

# construim datele de antrenare
training = []
output = []
output_empty = [0] * len(
    classes)  # un vector de 0 de lungime egala cu numarul de clase (vom folosi reprezentarea one-hot a claselor)

for doc in documents:  # bag of words pentru fiecare fraza
    bag = []

    pattern_words = doc[0]  # lista de tokeni pentru fraza curenta
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    for w in words:  # construim vectorul binar pentru bag of words
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(
        doc[1])] = 1  # output este '1' pentru tagul corespunzator frazei si '0' pentru celelalte (one-hot)

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])  # feature-urile de antrenare
train_y = list(training[:, 1])  # etichete/labels (ce dorim sa returneze modelul)

# creating model and training
ops.reset_default_graph()  # resetam starea engine-ului TensorFlow

net = tflearn.input_data(shape=[None, len(train_x[0])])  # toti vectorii de bag-of-words au aceasta dimensiune
net = tflearn.fully_connected(net, 64)  # , activation='ReLu'
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')  # numarul de noduri output = numarul de clase
net = tflearn.regression(net)  # folosim acest strat de logistic regression pentru a extrage probabilitatile claselor

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs', tensorboard_verbose=3)  # definim modelul final
model.fit(train_x, train_y, n_epoch=100, batch_size=8,
          show_metric=True)  # incepem antrenarea folosind coborarea pe gradient, trecem prin model cate 8 fraze odata
model.save('model.tflearn')

pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open("training_data", "wb"))

# testing
data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

model.load('model.tflearn')  # incarcam weight-urile salvate pentru cel mai bun model de clasificare


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)  # tokenizare
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]  # stemming

    return sentence_words


# reprezentare binara bag-of-words
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return np.array(bag)


ERROR_THRESHOLD = 0.4


def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]  # probabilitatile prezise de model
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]  # renuntam la intentiile improbabile
    results.sort(key=lambda x: x[1], reverse=True)  # sortam descrescator intentiile dupa probabilitate
    return_list = []

    for r in results:
        return_list.append((classes[r[0]], r[1]))

    return return_list  # return perechi (intentie, probabilitate)


# structura de date pentru context
context = {}


def response(sentence, userID='27'):
    results = classify(sentence)

    if results:  # daca avem macar o intentie valida, o procesam pe cea cu probabilitate maxima
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:  # cautam in dictionarul de intentii tagul returnat
                    if 'context_filter' not in i or (userID in context and 'context_filter' in i and i['context_filter']
                                                     == context[userID]):
                        if 'context_set' in i:  # daca intentia curenta actualizeaza contextul
                            context[userID] = i['context_set']
                        else:
                            context[userID] = {}

                        if i['tag'] == "weather":
                            html = weatherAcces().content
                            soup = bs4.BeautifulSoup(html, "html.parser")
                            submission_count_text = soup.find(class_="CurrentConditions--tempValue--3a50n").text
                            tts = gTTS(text=submission_count_text, lang='en', slow=False)
                            date_string = datetime.now().strftime("%d%m%Y%H%M%S")
                            ttsResponse = "tts." + date_string + ".mp3"
                            tts.save('tts/' + ttsResponse)
                            mixer.music.load('tts/' + ttsResponse)
                            mixer.music.play()
                            return submission_count_text, 'neutral'

                        botResponse = random.choice(i['responses'])  # returnam un raspuns corespunzator intentiei
                        tts = gTTS(text=botResponse, lang='en', slow=False)
                        date_string = datetime.now().strftime("%d%m%Y%H%M%S")
                        ttsResponse = "tts." + date_string + ".mp3"
                        tts.save('tts/' + ttsResponse)
                        mixer.music.load('tts/' + ttsResponse)
                        mixer.music.play()

                        if (i['tag'] == "greetingGood") or (i['tag'] == "compliment") or (i['tag'] == "feelingGood") \
                                or (i['tag'] == "approveJoke") or (i['tag'] == "joke") or (i['tag'] == "funny") or \
                                (i['tag'] == "thanks") or (i['tag'] == "goodbye"):
                            return botResponse, 'happy'
                        if (i['tag'] == "empty") or (i['tag'] == "greetingBad") or (i['tag'] == "feelingBad") or \
                                (i['tag'] == "disapproveJoke"):
                            return botResponse, 'sad'
                        return botResponse, 'neutral'

            results.pop(0)  # daca nu am putut da un raspuns, trecem la urmatoarea intentie cu probabilitate maxima

    # myobj = gTTS(text="Sorry, I can't understand you. :(", lang='en', slow=False)
    # myobj.save("notUnderstand.mp3")
    notUnderstand = 'notUnderstand.mp3'
    mixer.music.load(notUnderstand)
    mixer.music.play()
    return "Sorry, I can't understand you. :(", 'sad'  # nu a putut fi stabilita o intentie pentru fraza introdusa


def weatherAcces():
    return requests.get(f"https://weather.com/ro-RO/vreme/astazi/l/ROXX0003:1:RO")


if __name__ == "__main__":
    # text = get_audio()

    # we remove any previous tts files
    directory = 'tts/'
    for f in os.listdir(directory):
        os.remove(os.path.join(directory, f))

    g = GUI()
