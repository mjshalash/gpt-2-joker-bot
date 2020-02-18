import tkinter as tk
from tkinter import ttk, filedialog
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

        # Variables
        self.pretrainedModel = ''
        self.device = ''
        self.datasetFile = ''
        self.trainedModelFile = ''
        self.optimizor = ''
        self.scheduler = ''
        self.hyperBatch = ''
        self.hyperEpoch = ''
        self.hyperLearnRate = ''
        self.hyperWarmSteps = ''
        self.hyperTrainSteps = ''
        self.hyperMaxSeq = ''

        # Model Dropdown
        # TODO: Figure out how to set variable based on this
        self.modelDropLbl = tk.Label(master, text="Pretrained Model:")
        self.modelDrop = ttk.Combobox(master, values=["openai-gpt", "gpt2"])
        self.modelDropLbl.grid(column=0, row=0)
        self.modelDrop.grid(column=0, row=1)

        # Device Dropdown
        # TODO: Figure out how to set variable based on this
        self.deviceDropLbl = tk.Label(master, text="Device:")
        self.deviceDrop = ttk.Combobox(master, values=["cpu", "gpu"])
        self.deviceDropLbl.grid(column=0, row=2)
        self.deviceDrop.grid(column=0, row=3)

        # Dataset Browser
        self.datasetFinderLbl = tk.Label(master, text="Training Dataset:")
        self.datasetFinderTxt = Entry(
            master, textvariable=self.datasetFile)
        self.datasetFinderBtn = Button(
            master, text="Browse", command=self.findDatasetFile)
        self.datasetFinderLbl.grid(column=0, row=4)
        self.datasetFinderTxt.grid(column=0, row=5)
        self.datasetFinderBtn.grid(column=0, row=6)

        # Folder to Save Models
        self.saveModelFinderLbl = tk.Label(
            master, text="Trained Models Directory:")
        self.saveModelFinderTxt = Entry(
            master, textvariable=self.trainedModelFile)
        self.saveModelFinderBtn = Button(
            master, text="Browse", command=self.setSaveTrained)
        self.saveModelFinderLbl.grid(column=0, row=7)
        self.saveModelFinderTxt.grid(column=0, row=8)
        self.saveModelFinderBtn.grid(column=0, row=9)

        # TODO: Hyperparams, Optimizor, Scheduler

        # Optimizor
        self.optHeader = tk.Label(
            master, text="Optimizor")
        self.optTxt = Entry(
            master, textvariable=self.optimizor)

        # Schedule
        self.schedHeader = tk.Label(
            master, text="Scheduler", textvariable=self.scheduler)
        self.schedTxt = Entry(
            master)

        # Hyperparameters
        self.hyperHeader = tk.Label(
            master, text="Hyperparameters", bg='black', fg='white')

        self.hyperBatchLbl = tk.Label(master, text="Batch Size")
        self.hyperBatchTxt = Entry(
            master, textvariable=self.hyperBatch)

        self.hyperEpochLbl = tk.Label(master, text="Epochs")
        self.hyperEpochTxt = Entry(
            master, textvariable=self.hyperEpoch)

        self.hyperLearnLbl = tk.Label(master, text="Learning Rate")
        self.hyperLearnTxt = Entry(
            master, textvariable=self.hyperLearnRate)

        self.hyperWarmUpLbl = tk.Label(master, text="Warmup Steps")
        self.hyperWarmUpTxt = Entry(
            master, textvariable=self.hyperWarmSteps)

        self.hyperTrainingLbl = tk.Label(master, text="Training Steps")
        self.hyperTrainingTxt = Entry(
            master, textvariable=self.hyperTrainSteps)

        self.hyperMaxSeqLbl = tk.Label(master, text="Max Sequence")
        self.hyperMaxSeqTxt = Entry(
            master, textvariable=self.hyperMaxSeq)

        self.optHeader.grid(column=0, row=10)
        self.optTxt.grid(column=0, row=11)
        self.schedHeader.grid(column=0, row=12)
        self.schedTxt.grid(column=0, row=13)
        self.hyperHeader.grid(column=0, row=14)
        self.hyperBatchLbl.grid(column=0, row=15)
        self.hyperBatchTxt.grid(column=0, row=16)
        self.hyperEpochLbl.grid(column=0, row=17)
        self.hyperEpochTxt.grid(column=0, row=18)
        self.hyperLearnLbl.grid(column=0, row=19)
        self.hyperLearnTxt.grid(column=0, row=20)
        self.hyperWarmUpLbl.grid(column=0, row=21)
        self.hyperWarmUpTxt.grid(column=0, row=22)
        self.hyperTrainingLbl.grid(column=0, row=23)
        self.hyperTrainingTxt.grid(column=0, row=24)
        self.hyperMaxSeqLbl.grid(column=0, row=25)
        self.hyperMaxSeqTxt.grid(column=0, row=26)


def main():
    root = tk.Tk()
    root.geometry("1000x500")
    Window(root)
    root.wm_title("GPT-2 Joker")
    root.mainloop()


if __name__ == '__main__':
    main()
