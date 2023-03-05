from tkinter import *

from GUI.components.apply_frame import apply_frame
from GUI.components.model_frame import model_frame


def model_screen(master):
    m_frame = model_frame(master)
    a_frame = apply_frame(master)
    m_frame.grid(
        column=0,
        row=0,
        sticky=N+S+W+E
    )
    a_frame.grid(
        column=1,
        row=0,
        sticky=N+S+W+E
    )
