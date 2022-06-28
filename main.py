import threading
from tkinter import *
import warnings
import nltk
from django.utils.datetime_safe import datetime
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
from tensorflow.python.framework import ops
import tflearn
import random
import json
import pickle
from gtts import gTTS
import os
from pygame import mixer
import requests
import bs4
from time import *
import datetime as dt
import transformers


class GUI:
    def __init__(self):
        self.Window = Tk()
        self.Window.withdraw()
        self.screen_width = self.Window.winfo_screenwidth()
        self.screen_height = self.Window.winfo_screenheight()

        # login window
        self.login = Toplevel()
        self.login.title("Login")
        login_width, login_height = 400, 300
        self.login.resizable(width=False, height=False)
        self.login.configure(width=login_width, height=login_height)
        self.login.geometry(f'{login_width}x{login_height}+{int((self.screen_width - login_width)/2)}+{int((self.screen_height - login_height)/2)}')

        self.pls = Label(self.login, text="What should I call you?", justify=CENTER, font="Roboto 14 bold")
        self.pls.place(relheight=0.15, relx=0.2, rely=0.07)

        self.labelName = Label(self.login, text="Name: ", font="Roboto 12")
        self.labelName.place(relheight=0.2, relx=0.1, rely=0.2)

        # entry box for name
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
        Window_width, Window_height = 1200, 700
        self.Window.resizable(width=False, height=False)
        self.Window.configure(width=Window_width, height=Window_height, bg="#17202A")
        self.Window.geometry(f'{Window_width}x{Window_height}+{int((self.screen_width - Window_width)/2)}+{int((self.screen_height - Window_height)/2)}')

        width = 0.7

        # chat window
        self.chatWindow = Text(self.Window, width=20, height=2, bg="#17202A", fg="#EAECEE", font="Roboto 14", padx=5,
                               pady=5)
        self.chatWindow.place(relheight=0.825, relwidth=width)

        # avatar
        self.avatar = PhotoImage(file='avatar.png')  # need a reference to the image or it gets garbage collected
        self.avatarFrame = Label(self.Window, image=self.avatar)
        self.avatarFrame.place(relx=width, rely=0)

        # settings button
        self.buttonSettings = Button(self.Window, text="Settings", font="Roboto 14 bold", width=20,
                                     bg="#ABB2B9", command=lambda: self.openSettings())
        self.buttonSettings.place(relx=0.76, rely=0.5)
        self.chatWindow.config(cursor="arrow")

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
        elif feeling == "wink":
            self.avatar = PhotoImage(file='avatar_wink.png')
            self.avatarFrame.configure(image=self.avatar)
        elif feeling == "wow":
            self.avatar = PhotoImage(file='avatar_wow.png')
            self.avatarFrame.configure(image=self.avatar)
        else:
            self.avatar = PhotoImage(file='avatar.png')
            self.avatarFrame.configure(image=self.avatar)
            if feeling == "exit":
                self.Window.after(2000, lambda: self.Window.destroy())
        self.avatarFrame.image = self.avatar

    def openSettings(self):
        self.settings = Toplevel()
        self.settings.title("Color Settings")
        self.settings.resizable(width=False, height=False)
        self.settings.configure(width=400, height=300)

        self.buttonSettingsRed = Button(self.settings, text="Red Backround", font="Roboto 14 bold", width=20, bg="#ABB2B9", command=lambda: self.applySettings('red'))
        self.buttonSettingsRed.place(relx=0.22, rely=0.12)
        self.chatWindow.config(cursor="arrow")

        self.buttonSettingsYellow = Button(self.settings, text="Yellow Backround", font="Roboto 14 bold", width=20, bg="#ABB2B9", command=lambda: self.applySettings('yellow'))
        self.buttonSettingsYellow.place(relx=0.22, rely=0.32)
        self.chatWindow.config(cursor="arrow")

        self.buttonSettingsGreen = Button(self.settings, text="Green Backround", font="Roboto 14 bold", width=20, bg="#ABB2B9", command=lambda: self.applySettings('green'))
        self.buttonSettingsGreen.place(relx=0.22, rely=0.52)
        self.chatWindow.config(cursor="arrow")

        self.buttonSettingsDefault = Button(self.settings, text="Default Backround", font="Roboto 14 bold", width=20, bg="#ABB2B9", command=lambda: self.applySettings('blue'))
        self.buttonSettingsDefault.place(relx=0.22, rely=0.72)
        self.chatWindow.config(cursor="arrow")

    def applySettings(self, color):
        if color == 'red':
            self.chatWindow.configure(bg="#b03333")  # roșu
            self.Window.configure(bg="#b03333")
            self.textbox.configure(bg="#b03333")
            self.settings.destroy()
            return
        elif color == 'yellow':
            self.chatWindow.configure(bg="#757523")  # galben
            self.Window.configure(bg="#757523")
            self.textbox.configure(bg="#757523")
            self.settings.destroy()
            return
        elif color == 'green':
            self.chatWindow.configure(bg="#19542e")  # verde
            self.Window.configure(bg="#19542e")
            self.textbox.configure(bg="#19542e")
            self.settings.destroy()
            return
        self.chatWindow.configure(bg="#17202A")  # albastru
        self.Window.configure(bg="#17202A")
        self.textbox.configure(bg="#2C3E50")
        self.settings.destroy()
        return


warnings.filterwarnings('ignore')
stemmer = LancasterStemmer()

with open('intents.json') as json_data:
    intents = json.load(json_data)

# tokenizing words
words = []
tags = []
wordTag = []
ignore_words = ['?', "'", ',', '.', '!']  # STOPWORDS

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)  # adaugam cuvintele la o lista globala cu toate cuvintele din texte
        wordTag.append((w, intent['tag']))  # adaugam textul tokenizat la multimea de documente
        if intent['tag'] not in tags:  # adaugam clasa (tipul de intentie) la lista de clase existente
            tags.append(intent['tag'])

# aplicam stemming si lowercasing pentru fiecare cuvant intalnit, ignoram stopwords
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))  # eliminam duplciate
tags = sorted(list(set(tags)))

# construim datele de antrenare
training = []
output = []
output_empty = [0] * len(
    tags)  # un vector de 0 de lungime egala cu numarul de clase (vom folosi reprezentarea one-hot a claselor)

for doc in wordTag:
    bag = []  # bag of words pentru fiecare fraza

    pattern_words = doc[0]  # lista de tokeni pentru fraza curenta
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    for w in words:  # construim vectorul binar pentru bag of words
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[tags.index(
        doc[1])] = 1  # output este '1' pentru tagul corespunzator frazei si '0' pentru celelalte (one-hot)

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])  # feature-urile de antrenare
train_y = list(training[:, 1])  # etichete/labels (ce dorim sa returneze modelul)

# creating model and training
ops.reset_default_graph()  # resetam starea engine-ului TensorFlow

net = tflearn.input_data(shape=[None, len(train_x[0])])  # toti vectorii de bag-of-words au aceasta dimensiune
net = tflearn.fully_connected(net, 64, activation='ReLu')
net = tflearn.fully_connected(net, 32, activation='ReLu')
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')  # numarul de noduri output = numarul de clase
net = tflearn.regression(net)  # folosim acest strat de logistic regression pentru a extrage probabilitatile claselor

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs', tensorboard_verbose=3)  # definim modelul final
model.fit(train_x, train_y, n_epoch=60, batch_size=8, show_metric=True)  # incepem antrenarea folosind coborarea pe gradient, trecem prin model cate 8 fraze odata
model.save('model.tflearn')

pickle.dump({'words': words, 'tags': tags, 'train_x': train_x, 'train_y': train_y}, open("training_data", "wb"))

# the bot
data = pickle.load(open("training_data", "rb"))
words = data['words']
tags = data['tags']
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


ERROR_THRESHOLD = 0.6


def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]  # probabilitatile prezise de model
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]  # renuntam la intentiile improbabile
    results.sort(key=lambda x: x[1], reverse=True)  # sortam descrescator intentiile dupa probabilitate
    return_list = []

    for r in results:
        return_list.append((tags[r[0]], r[1]))

    return return_list  # return perechi (intentie, probabilitate)


# structura de date pentru context
context = {}


def response(sentence, userID='27'):
    if sentence == '' or sentence == ' ':
        botResponse = random.choice(['Did you say something?', 'Are you there?', 'Hello?'])
        tts = gTTS(text=botResponse, lang='en', slow=False)
        date_string = datetime.now().strftime("%d%m%Y%H%M%S")
        ttsResponse = "tts." + date_string + ".mp3"
        tts.save('tts/' + ttsResponse)
        mixer.music.load('tts/' + ttsResponse)
        mixer.music.play()
        return botResponse, 'sad'

    results = classify(sentence)

    if results:  # daca avem macar o intentie valida, o procesam pe cea cu probabilitate maxima
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:  # cautam in dictionarul de intentii tagul returnat
                    if 'context_filter' not in i or (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if 'context_set' in i:  # daca intentia curenta actualizeaza contextul
                            context[userID] = i['context_set']
                        else:
                            context[userID] = {}

                        if i['tag'] == "goodbye":
                            botResponse = random.choice(i['responses'])
                            tts = gTTS(text=botResponse, lang='en', slow=False)
                            date_string = datetime.now().strftime("%d%m%Y%H%M%S")
                            ttsResponse = "tts." + date_string + ".mp3"
                            tts.save('tts/' + ttsResponse)
                            mixer.music.load('tts/' + ttsResponse)
                            mixer.music.play()
                            return botResponse, 'exit'

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

                        if i['tag'] == "today":
                            date = dt.datetime.now()
                            date = str(date)
                            tts = gTTS(text=date, lang='en', slow=False)
                            date_string = datetime.now().strftime("%d%m%Y%H%M%S")
                            ttsResponse = "tts." + date_string + ".mp3"
                            tts.save('tts/' + ttsResponse)
                            mixer.music.load('tts/' + ttsResponse)
                            mixer.music.play()
                            return date, 'neutral'

                        botResponse = random.choice(i['responses'])  # returnam un raspuns corespunzator intentiei
                        tts = gTTS(text=botResponse, lang='en', slow=False)
                        date_string = datetime.now().strftime("%d%m%Y%H%M%S")
                        ttsResponse = "tts." + date_string + ".mp3"
                        tts.save('tts/' + ttsResponse)
                        mixer.music.load('tts/' + ttsResponse)
                        mixer.music.play()

                        if (i['tag'] == "greetingGood") or (i['tag'] == "compliment") or (i['tag'] == "feelingGood") or (i['tag'] == "approveJoke") or (i['tag'] == "joke") or (i['tag'] == "funny") or (i['tag'] == "thanks") or (i['tag'] == "botNeed") or (i['tag'] == "botFeeling") or (i['tag'] == "botFriend") or (i['tag'] == "cute") or (i['tag'] == "welcome"):
                            return botResponse, 'happy'
                        if (i['tag'] == "greetingBad") or (i['tag'] == "feelingBad") or (i['tag'] == "notFunny") or (i['tag'] == "disapproveJoke") or (i['tag'] == "feelingBad") or (i['tag'] == "botAbilityDance") or (i['tag'] == "help"):
                            return botResponse, 'sad'
                        if (i['tag'] == "greetingStarWars") or (i['tag'] == "greetingMissing") or (i['tag'] == "botCall") or (i['tag'] == "botNameMeaning") or (i['tag'] == "botNameWeird") or (i['tag'] == "botCount") or (i['tag'] == "botScary"):
                            return botResponse, 'wink'
                        if i['tag'] == "botSingNegative":
                            return botResponse, 'wow'
                        return botResponse, 'neutral'

            results.pop(0)  # daca nu am putut da un raspuns, trecem la urmatoarea intentie cu probabilitate maxima

    chat = nlp(transformers.Conversation(sentence), pad_token_id=50256)
    res = str(chat)
    res = res[res.find("bot >> ") + 6:].strip()

    tts = gTTS(text=res, lang='en', slow=False)
    date_string = datetime.now().strftime("%d%m%Y%H%M%S")
    ttsResponse = "tts." + date_string + ".mp3"
    tts.save('tts/' + ttsResponse)
    mixer.music.load('tts/' + ttsResponse)
    mixer.music.play()

    return res, 'neutral'


def weatherAcces():
    return requests.get(f"https://weather.com/ro-RO/vreme/astazi/l/ROXX0003:1:RO")


if __name__ == "__main__":

    nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # we remove any previous tts files
    directory = 'tts/'
    for f in os.listdir(directory):
        os.remove(os.path.join(directory, f))

    g = GUI()
