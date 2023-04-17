from tkinter.ttk import *


def styling(master):
    style = Style(master)
    default_font = master.option_get("font", "TkDefaultFont")
    title_font = (default_font, 11)

    # shaded
    style.configure('ShadedFrame.TFrame', background='lightgray')
    style.configure('ShadedTitle.TLabel', font=title_font, background="lightgray")
    style.configure('ShadedText.TLabel', background="lightgray")
    style.configure('ShadedButton.TButton', background='lightgray')

    # text
    style.configure('Title.TLabel', font=title_font)
