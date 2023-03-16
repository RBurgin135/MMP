from tkinter.ttk import *
from tkinter import filedialog


class ModelButtonFrame(LabelFrame):
    def __init__(self, master, controller, current_model, *args, **kwargs):
        super().__init__(master, text="button frame", style='Custom.TFrame', *args, **kwargs)

        # filesystem info
        filetypes = (
            ('Tensorflow Files', '*.tf'),
            ('All files', '*.*')
        )
        initial_dir = ""

        # content
        new = Button(
            master=self,
            text="New",
            command=lambda: controller.navigate("training")
        )
        save = Button(
            master=self,
            text="Save",
            command=lambda: filedialog.asksaveasfilename(
                title="Save a model",
                initialdir=initial_dir,
                filetypes=filetypes
            ),
            state="normal" if current_model.has_data else"disabled"
        )
        load = Button(
            master=self,
            text="Load",
            command=lambda: filedialog.askopenfilename(
                title="Load a model",
                initialdir=initial_dir,
                filetypes=filetypes
            )
        )
        new.grid(column=0, row=0, padx=5, sticky="wen")
        save.grid(column=1, row=0, padx=5, sticky="wen")
        load.grid(column=2, row=0, padx=5, sticky="wen")
        self.rowconfigure(0, weight=1)
        for c in range(3):
            self.columnconfigure(c, weight=1)
