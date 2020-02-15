import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import *


class Window(Frame):
    # TODO: Can probably combine file finder methods
    # Method to browse for training dataset file
    def findDatasetFile(self):
        self.datasetFinderTxt.insert(INSERT, filedialog.askopenfilename(
            initialdir="D:\Malik Shalash\Documents\Development\School\CECS-696\gpt-2-joker-bot\datasets", title="Select Training Data File"))

    # Method to browse for training dataset file
    def setSaveTrained(self):
        self.saveModelFinderTxt.insert(INSERT, filedialog.askdirectory(
            initialdir="D:\Malik Shalash\Documents\Development\School\CECS-696\gpt-2-joker-bot", title="Select Directory for Trained Models"))

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master

        # Model Dropdown
        self.modelDropLbl = tk.Label(master, text="Pretrained Model:")
        self.modelDrop = ttk.Combobox(master, values=["openai-gpt", "gpt2"])
        self.modelDropLbl.grid(column=0, row=0)
        self.modelDrop.grid(column=0, row=1)

        # Device Dropdown
        self.deviceDropLbl = tk.Label(master, text="Device:")
        self.deviceDrop = ttk.Combobox(master, values=["cpu", "gpu"])
        self.deviceDropLbl.grid(column=0, row=2)
        self.deviceDrop.grid(column=0, row=3)

        # Dataset Browser

        self.datasetFinderLbl = tk.Label(master, text="Training Dataset:")
        self.datasetFinderTxt = Text(
            master, bg="#FFFFFF", fg="#000000", height=1)
        self.datasetFinderBtn = Button(
            master, text="Browse", command=self.findDatasetFile)
        self.datasetFinderLbl.grid(column=0, row=4)
        self.datasetFinderTxt.grid(column=0, row=5)
        self.datasetFinderBtn.grid(column=0, row=6)

        # Folder to Save Models
        self.saveModelFinderLbl = tk.Label(
            master, text="Trained Models Directory:")
        self.saveModelFinderTxt = Text(
            master, bg="#FFFFFF", fg="#000000", height=1)
        self.saveModelFinderBtn = Button(
            master, text="Browse", command=self.setSaveTrained)
        self.saveModelFinderLbl.grid(column=0, row=7)
        self.saveModelFinderTxt.grid(column=0, row=8)
        self.saveModelFinderBtn.grid(column=0, row=9)

        # TODO: Hyperparams, Optimizor, Scheduler

        # Example for textfield
        # self.resultField = Text(master, bg="#FFFFFF", fg="#000000", height=1)
        # self.resultField.insert(INSERT, "0")
        # self.resultField.grid(row=5, columnspan=4)


root = tk.Tk()
root.geometry("1000x500")
app = Window(root)
root.wm_title("GPT-2 Joker")
root.mainloop()
