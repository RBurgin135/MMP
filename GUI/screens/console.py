from tkinter.ttk import *
import tkinter as tk

widget = None


class Console:

    @staticmethod
    def initialise_widget(master):
        global widget
        widget = tk.Text(
            master=master,
            state='disabled',
            bg='white', fg='black', bd=0
        )
        return widget

    @staticmethod
    def print(text):
        widget.configure(state='normal')
        widget.insert('end', f"  {text}\n")
        widget.yview('end')
        widget.configure(state='disabled')
