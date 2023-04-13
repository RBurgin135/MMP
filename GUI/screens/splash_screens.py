from tkinter.ttk import *


class LoadingScreen(Frame):
    def __init__(self, master, controller, current_model, **kwargs):
        super().__init__(master, **kwargs)

        # content
        Label(
            master=self,
            text="Loading..."
        ).pack(
            pady=25
        )


class SavingScreen(Frame):
    def __init__(self, master, controller, current_model, **kwargs):
        super().__init__(master, **kwargs)

        # content
        Label(
            master=self,
            text="Saving..."
        ).pack(
            pady=25
        )
