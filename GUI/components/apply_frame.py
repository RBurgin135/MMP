import tkinter as tk
from tkinter.ttk import *
from tkinter import filedialog


def apply_frame(master, is_data):
    frame = LabelFrame(
        master=master,
        text="This is apply frame",
        padding=10
    )
    title = Label(
        master=frame,
        text="Apply Model"
    )
    title.pack(
        padx=10,
        pady=10
    )
    dataset_button = file_path_button(frame, "Dataset path")
    dataset_button.pack()
    output_button = file_path_button(frame, "Output path")
    output_button.pack()
    type_button = output_type_button(frame)
    type_button.pack()
    return frame


def file_path_button(master, text):
    file_button = Button(
        master=master,
        text=text,
        width=100,
        command=lambda: filedialog.askopenfilename(
            initialdir="/",
            title=text,
            filetypes=(("python files", "*.py"), ("all files", "*.*"))
        )
    )
    return file_button


def output_type_button(master):
    options = [
        "Output type",
        "Segmentation mask",
        "Point map",
        "etc"
    ]
    clicked = tk.StringVar()
    clicked.set("Segmentation mask")
    output = OptionMenu(
        master,
        clicked,
        *options,
        command=lambda x: print(x)
    )
    return output
