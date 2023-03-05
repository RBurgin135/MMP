import tkinter as tk
from tkinter.ttk import *
from tkinter.constants import *


def model_frame(master):
    # components
    m_frame = LabelFrame(
        master=master,
        text="This is model frame",
        padding=10
    )
    title = model_title(m_frame, True)
    info = Label(
        master=m_frame,
        text="Lorem ipsum dolor sit am"
    )

    # place
    title.grid(
        column=0,
        row=0,
        sticky=N
    )
    info.grid(
        column=0,
        row=1,
        sticky=N + W
    )
    b_frame = button_frame(m_frame)
    b_frame.grid(
        column=0,
        row=2,
        sticky=E + W + S
    )
    m_frame.rowconfigure(2, weight=1)
    m_frame.columnconfigure(0, weight=1)
    return m_frame


def button_frame(master):
    frame = LabelFrame(
        master=master,
        text="Button frame",
        padding=10
    )
    new = Button(
        master=frame,
        text="New",
        command=lambda: print("new")
    )
    save = Button(
        master=frame,
        text="Save",
        command=lambda: print("save")
    )
    load = Button(
        master=frame,
        text="Load",
        command=lambda: print("load")
    )
    new.grid(column=0, row=0, padx=5, sticky=W + E + N)
    save.grid(column=1, row=0, padx=5, sticky=W + E + N)
    load.grid(column=2, row=0, padx=5, sticky=W + E + N)
    frame.rowconfigure(0, weight=1)
    for c in range(3):
        frame.columnconfigure(c, weight=1)
    return frame


def model_title(master, is_data):
    if is_data:
        return Label(
            master=master,
            text="No Data Stored",
        )
    else:
        return Label(
            master=master,
            text="Model Name"
        )
