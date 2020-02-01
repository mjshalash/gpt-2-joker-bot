from transformers import AdamW, get_linear_schedule_with_warmup  # WarmupLinearSchedule
from torch.utils.data import Dataset
import csv
import json
import os
from torch.utils.data import Dataset, DataLoader
import logging
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import warnings
warnings.filterwarnings('ignore')

logging.getLogger().setLevel(logging.CRITICAL)

# Hyperparameters for Model Training
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000
TRAINING_STEPS = 10000
MAX_SEQ_LEN = 400

# If CUDA GPU available, use it
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    print("GPU detected, utilizing GPU")


# TODO: Make model name a variable to automate organizing saved models
# Retrieve language model from huggingface AWS S3 Bucket

# Byte-pair encoder
# Transforms input text into recognizable input tokens
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# GPT2LMHeadModel is the GPT2Model with an extra linear layer
# Extra layer does inverse op of embedding layer to create dict. of possible word outcomes
model = GPT2LMHeadModel.from_pretrained('gpt2')

print("Model Imported!")

# Custom dataset class which inherits abstract Dataset (from pytorch)


class JokesDataset(Dataset):
    def __init__(self, jokes_dataset_path='data/short_jokes/'):
        # Find joke dataset
        super().__init__()

        short_jokes_path = os.path.join(
            jokes_dataset_path, 'shortjokes.csv')

        # Concatenate <|endoftext\> to end of jokes
        # Recognized by GPT-2 as end of token marker
        # Allows for multiple jokes in one sequence
        self.joke_list = []
        self.end_of_text_token = "<|endoftext|>"

        # Strip joke from each line of csv
        with open(short_jokes_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            x = 0
            for row in csv_reader:
                joke_str = f"JOKE:{row[1]}{self.end_of_text_token}"
                self.joke_list.append(joke_str)
        # print(len(self.joke_list))

    # Overrides of abstract Dataset class
    def __len__(self):
        return len(self.joke_list)

    def __getitem__(self, item):
        return self.joke_list[item]


# Create Dataset object
dataset = JokesDataset()

# DataLoader is a better iterator than simple for-loop
# Gives access to Batching data, Shuffling data, and use of multiprocessing workers
joke_loader = DataLoader(dataset, batch_size=1, shuffle=True)

#################### Train Model #############################
# Train the model and save the model weights after each epoch
# Generate jokes with each version of weight to see what is best

# Move model to specified device for training
model = model.to(device)
model.train()

# Optimizer to determine how, when and by what model parameters are updated
# Update model params based on loss function results
# Adam = adaptive moment estimation = uses past gradients to calculate current
# aka momentum
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# How the learning rate (magnitude of weight change) adapts over time
# Linear with warmup = HIGH learning rate at the beginning and then stays constant
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=TRAINING_STEPS,  last_epoch=-1)


proc_seq_count = 0
sum_loss = 0.0
batch_count = 0

tmp_jokes_tens = None

# Establish folder to save trained models
models_folder = "trained_models"
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

# Every Epoch
for epoch in range(EPOCHS):

    print(f"EPOCH {epoch} started" + '=' * 30)

    # Enumerate gives access to counter (idx) and item (joke)
    for idx, joke in enumerate(joke_loader):

        #################### "Fit as many joke sequences into MAX_SEQ_LEN sequence as possible" logic start ####
        # tensor = multidimensional array with variable num. of axis
        # Vector = 1-order tensor, Matrix = 2 order tensor
        # Creates tensor with joke data
        # Tokenizer.encode is encoding joke input
        joke_tens = torch.tensor(tokenizer.encode(
            joke[0])).unsqueeze(0).to(device)

        # Skip sample from dataset if it is longer than MAX_SEQ_LEN
        if joke_tens.size()[1] > MAX_SEQ_LEN:
            continue

        # The first joke sequence in the sequence
        if not torch.is_tensor(tmp_jokes_tens):
            tmp_jokes_tens = joke_tens
            continue
        else:
            # The next joke does not fit in so we process the sequence and leave the last joke
            # as the start for next sequence
            if tmp_jokes_tens.size()[1] + joke_tens.size()[1] > MAX_SEQ_LEN:
                work_jokes_tens = tmp_jokes_tens
                tmp_jokes_tens = joke_tens
            else:
                # Add the joke to sequence, continue and try to add more
                tmp_jokes_tens = torch.cat(
                    [tmp_jokes_tens, joke_tens[:, 1:]], dim=1)
                continue

        ################## Sequence ready, process it trough the model ##################

        outputs = model(work_jokes_tens, labels=work_jokes_tens)
        loss, logits = outputs[:2]

        # .backward() is a part of Autograd
        # Autograd is the automatic differentiation library for pytorch
        # This performs backpropagation
        loss.backward()
        sum_loss = sum_loss + loss.detach().data

        # Increment sequence counter
        proc_seq_count = proc_seq_count + 1

        # If we have completed 1 batch (a certain number of sequences)
        # Please note important step details below
        if proc_seq_count == BATCH_SIZE:
            proc_seq_count = 0
            batch_count += 1
            optimizer.step()        # This is important detail, we do not take learning step
            scheduler.step()        # after ALL training data, we do it after each batch
            optimizer.zero_grad()
            model.zero_grad()

        # Print output every 100th batch
        if batch_count == 100:
            print(f"sum loss {sum_loss}")
            batch_count = 0
            sum_loss = 0.0

    # Store the model after each epoch to compare the performance of them
    torch.save(model.state_dict(), os.path.join(
        models_folder, f"gpt2_joker_{epoch}.pt"))
