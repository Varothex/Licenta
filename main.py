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
        message = "Varothex: Hello there, my name is Varothex. I will answer your queries. If you want to exit, " \
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
        message = bot_response(self.msg)
        self.textCons.config(state=NORMAL)
        self.textCons.insert(END, message + "\n\n")
        self.textCons.config(state=DISABLED)
        self.textCons.see(END)


warnings.filterwarnings('ignore')

f = open('C:\\Users\\mihai\\Desktop\\Serios\\Projects\\Python\\Bot\\robot.txt', 'r', errors='ignore')
raw = f.read()
raw = raw.lower()

sent_tokens = nltk.sent_tokenize(raw)  # converts to list of scentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words

sentToken = sent_tokens[:4]
wordToken = word_tokens[:4]

# preprocessing
lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["Hi!", "Hey!", "*nods*", "Hello there!", "Hello!"]


def greeting(scentence):
    for word in scentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response):
    chatbot_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        chatbot_response = chatbot_response + "I am sorry, I didn't understand you. :("
        return chatbot_response

    else:
        chatbot_response = chatbot_response + sent_tokens[idx]
        return chatbot_response


def bot_response(user_response):
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response.lower() == 'thanks' or user_response == 'thank you':
            # print("Varothex: You're welcome!")
            return "Varothex: You're welcome!"
        else:
            if greeting(user_response) is not None:
                # print("Varothex: " + greeting(user_response))
                return "Varothex: " + greeting(user_response)
            else:
                # print("Varothex: ", end='')
                # print(response(user_response))
                # sent_tokens.remove(user_response)
                return "Varothex: " + response(user_response)
                # sent_tokens.remove(user_response)
    else:
        # print("Varothex: Bye!")
        return "Varothex: Bye!"


if __name__ == "__main__":
    g = GUI()
