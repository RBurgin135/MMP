import tkinter as tk
import tkinter.ttk as ttk
from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1)
from GUI.model_frame import model_frame

root = tk.Tk()
root.title("PCA Wavelet Model Manager")
# root.iconbitmap('GUI/assets/icon.ico')
root.iconbitmap('GUI/assets/kingfisher.ico')


model_frame(root).pack()

'''greeting = tk.Label(
    text="Hello, Tkinter",
    foreground="purple",  # Set the text color to white
    background="yellow"  # Set the background color to black
)
themedGreeting = ttk.Label(text="Hello World")
greeting.pack()
themedGreeting.pack()

button = tk.Button(
    text="hi!",
    width=25,
    height=5,
    bg="blue",
    fg="yellow",
)
buttonTtk = ttk.Button(
    text="hi!",
    width=15
)
button.pack()
buttonTtk.pack()

entry = tk.Entry(fg="yellow", bg="blue", width=50)
entryTtk = ttk.Entry(width=50)
entry.pack()
entryTtk.pack()
'''




if __name__ == "__main__":
    root.mainloop()
