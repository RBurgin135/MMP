from tkinter.ttk import *


class TopNavigationBar(LabelFrame):
    def __init__(self, master, controller, title, nav_button=None, *args, **kwargs):
        super().__init__(master, text="top navigation bar", *args, **kwargs)

        # content
        # navigation button
        if nav_button:
            Button(
                master=self,
                text=nav_button.text,
                image=nav_button.image,
                command=nav_button.command
            ).grid(
                column=0,
                row=0,
                padx=10
            )

        # title
        Label(
            master=self,
            text=title
        ).grid(
            column=1,
            row=0
        )


class NavButtonData:
    def __init__(self, text=None, image=None, command=None):
        self.text = text
        self.image = image
        self.command = command
