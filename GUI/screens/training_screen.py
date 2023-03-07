from tkinter import *

from GUI.components.top_nav_bar import TopNavigationBar, NavButtonData


class TrainingScreen(LabelFrame):
    def __init__(self, master, controller):
        super().__init__(master, text="training screen")
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        # content
        TopNavigationBar(
            master=self,
            controller=controller,
            title="Train New Model",
            nav_button=NavButtonData(
                text="back",
                command=lambda: controller.navigate("model")
            )
        ).grid(
            column=0,
            row=0,
            sticky="ew"
        )
