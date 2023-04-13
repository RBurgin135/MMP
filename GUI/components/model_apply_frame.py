import tkinter as tk
from tkinter.ttk import *

from GUI.components.path_frame import PathFrame
from GUI.components.util import SetToDefaultsButton


class ModelApplyFrame(LabelFrame):
    def __init__(self, master, controller, current_model, **kwargs):
        super().__init__(master, text="apply frame", **kwargs)
        content = LabelFrame(master=self, text="content")
        content.pack()

        # content
        # title
        Label(
            master=content,
            text="Apply Model"
        ).pack(
            padx=5,
            pady=15,
            anchor="n"
        )

        if current_model.pca_wavelet_model is None:
            # no data image
            self.no_data_image = tk.PhotoImage(file="GUI/assets/no_data.png")
            Label(
                master=content,
                image=self.no_data_image,
                width=1
            ).pack(
                anchor="n",
                padx=5,
                pady=10
            )
            # no data message
            Label(
                master=content,
                text="Load or train a model to begin"
            ).pack(
                anchor="n",
                padx=5,
                pady=10
            )
        else:
            # dataset button
            dataset_path = tk.StringVar()
            PathFrame(
                master=content,
                controller=controller,
                text="Dataset Path: ",
                text_variable=dataset_path
            ).pack(
                padx=10,
                pady=3,
                anchor="w"
            )

            # output path button
            output_path = tk.StringVar()
            PathFrame(
                master=content,
                controller=controller,
                text="Output Path: ",
                text_variable=output_path
            ).pack(
                padx=10,
                pady=3,
                anchor="w"
            )

            # action button frame
            ButtonFrame(
                master=self,
                controller=controller,
                variables=[
                    dataset_path,
                    output_path
                ]
            ).pack(
                side="bottom",
                fill="both"
            )

class ButtonFrame(LabelFrame):
    def __init__(self, master, controller, variables, **kwargs):
        super().__init__(master, text="button frame", **kwargs)
        content = LabelFrame(self, text="content")
        content.pack()

        # content
        # set to defaults button
        SetToDefaultsButton(
            master=content,
            variables=variables
        ).pack(side='left')

        # train button
        Button(
            master=content,
            text="Begin",
            command=lambda: controller.navigate("process")
        ).pack(side='left')


class OutputTypeFrame(Frame):
    def __init__(self, text_variable, **kwargs):
        super().__init__(**kwargs)

        # options
        options = [
            "",
            "Segmentation mask",
            "Point map",
            "etc"
        ]
        text_variable.set(options[0])

        # content
        # label
        Label(
            master=self,
            text="Output Type:"
        ).pack(side='left')
        # option menu
        OptionMenu(
            self,
            text_variable,
            *options,
            command=lambda x: print(x)
        ).pack(side='left')

