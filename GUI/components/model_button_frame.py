from tkinter.ttk import *


class ModelButtonFrame(LabelFrame):
    def __init__(self, master, controller):
        super().__init__(master, text="button frame")

        # content
        new = Button(
            master=self,
            text="New",
            command=lambda: controller.navigate("training")
        )
        save = Button(
            master=self,
            text="Save",
            command=lambda: print("save")
        )
        load = Button(
            master=self,
            text="Load",
            command=lambda: print("load")
        )
        new.grid(column=0, row=0, padx=5, sticky="wen")
        save.grid(column=1, row=0, padx=5, sticky="wen")
        load.grid(column=2, row=0, padx=5, sticky="wen")
        self.rowconfigure(0, weight=1)
        for c in range(3):
            self.columnconfigure(c, weight=1)
