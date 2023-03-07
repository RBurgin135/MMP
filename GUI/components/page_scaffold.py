from tkinter import *


class PageScaffold(Frame):
    def __init__(self, master, controller, top, content):
        Frame.__init__(self, master)

        top.pack()
        content.pack()


class TopBar(Frame):
    def __init__(self, master, controller):
        Frame.__init__(self, master)
