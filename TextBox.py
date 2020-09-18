from tkinter import *
import WebScraper

webby = WebScraper


master = Tk()
e = Entry(master, width = 50)
e.pack()

e.focus_set()

def callback():
    url = e.get()# This is the text you may want to use later
    webby.highlightWebsite(url)
    e.delete(0, 1000)

b = Button(master, text = "OK", width = 50, height = 30, command = callback)
b.pack()

mainloop()

