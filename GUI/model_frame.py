import tkinter as tk


def model_frame(master):
    m_frame = tk.LabelFrame(
        master=master,
        text="This is model frame",
        padx=10,
        pady=10)
    m_frame.pack(padx=20, pady=20)
    title = model_title(m_frame, True)
    title.pack()
    info = tk.Label(
        master=m_frame,
        text="Lorem ipsum dolor sit am")
    info.pack()
    b_frame = button_frame(m_frame)
    b_frame.pack()
    return m_frame


def button_frame(master):
    frame = tk.LabelFrame(
        master=master,
        text="Button frame",
        padx=10,
        pady=10)
    new = tk.Button(
        master=frame,
        text="New")
    save = tk.Button(
        master=frame,
        text="Save")
    load = tk.Button(
        master=frame,
        text="Load")
    new.grid(column=0, row=0, padx=5)
    save.grid(column=1, row=0, padx=5)
    load.grid(column=2, row=0, padx=5)
    return frame


def model_title(master, data):
    if data:
        return tk.Label(
            master=master,
            text="No Data Stored")
    else:
        return tk.Label(
            master=master,
            text="Model Name")

