from tkinter import *
from GUI.components.model_frame import model_frame


def model_screen(master):
    m_frame = model_frame(master)
    a_frame = apply_frame(master)
    m_frame.grid(column=0, row=0, sticky=N+E+S+W)
    a_frame.grid(column=1, row=0, sticky=N+E+S+W)


def apply_frame(master):
    frame = LabelFrame(
        master=master,
        text="This is apply frame",
        padx=10,
        pady=10)
    title = Label(
        master=frame,
        text="Apply Model")
    title.pack(padx=10, pady=10)
    return frame
