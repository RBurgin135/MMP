from tkinter import *


def model_frame(master):
    m_frame = LabelFrame(
        master=master,
        text="This is model frame",
        padx=10,
        pady=10)
    title = model_title(m_frame, True)
    title.pack()
    info = Label(
        master=m_frame,
        text="Lorem ipsum dolor sit am")
    info.pack()
    b_frame = button_frame(m_frame)
    b_frame.pack()
    return m_frame


def button_frame(master):
    frame = LabelFrame(
        master=master,
        text="Button frame",
        padx=10,
        pady=10)
    new = Button(
        master=frame,
        text="New",
        command=lambda: print("new"))
    save = Button(
        master=frame,
        text="Save",
        command=lambda: print("save"))
    load = Button(
        master=frame,
        text="Load",
        command=lambda: print("load"))
    new.grid(column=0, row=0, padx=5)
    save.grid(column=1, row=0, padx=5)
    load.grid(column=2, row=0, padx=5)
    return frame


def model_title(master, data):
    if data:
        return Label(
            master=master,
            text="No Data Stored",
        )
    else:
        return Label(
            master=master,
            text="Model Name")
