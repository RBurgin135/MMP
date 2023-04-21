from tkinter.font import Font
import tkinter as tk

widget = None


class Console:

    @staticmethod
    def initialise_widget(master, width, height):
        # make widget
        global widget
        widget = tk.Text(
            master=master,
            state='disabled',
            bg='white',
            fg='black',
            bd=0,
            font=Font(family='Consolas', size=10, weight='normal'),
            width=width,
            height=height
        )

        return widget

    @staticmethod
    def write(text):
        widget.configure(state='normal')
        widget.insert('end', f"  {text}\n")
        widget.yview('end')
        widget.configure(state='disabled')
