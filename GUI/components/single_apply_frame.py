from tkinter.ttk import *


class SingleApplyFrame(Frame):
    def __init__(self, master, controller, current_model, button_frame):
        super().__init__(master)

        # title
        Label(
            master=self,
            text="Apply Model to Image"
        ).pack(
            padx=5,
            pady=15,
            anchor="n"
        )

        button_frame(
            master=self,
            controller=controller,
            current_model=current_model,
            variables=[
            ]
        ).pack(
            side="bottom",
            fill="both"
        )
