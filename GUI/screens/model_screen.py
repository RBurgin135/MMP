from tkinter import *

from GUI.components.apply_frame import apply_frame
from GUI.components.model_frame import model_frame


def model_screen(master):
    is_data = False
    m_frame = model_frame(master, is_data)
    a_frame = apply_frame(master, is_data)
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
