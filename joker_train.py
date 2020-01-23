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

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000
TRAINING_STEPS = 10000
MAX_SEQ_LEN = 400

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    print("GPU detected, utilizing GPU")


# Make model name a variable to automate organizing saved models
# Options: gpt2, gpt2-medium, gpt2-large, gpt2=xl
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

print("Model Imported!")


class JokesDataset(Dataset):
    def __init__(self, jokes_dataset_path='data/short_jokes/'):
        super().__init__()

        short_jokes_path = os.path.join(
            jokes_dataset_path, 'shortjokes.csv')

        # Concatenate <|endoftext\> to end of jokes
        # Recognized by GPT-2 as end of token marker
        # Allows for multiple jokes in one sequence
        self.joke_list = []
        self.end_of_text_token = "<|endoftext|>"

        with open(short_jokes_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            x = 0
            for row in csv_reader:
                # innaprop_flag = 0
                # words = row[1].split(" ")

                ## Parse through each word in row ##
                # Check for innappropriate words in kaggle dataset
                # for word in words:
                #   if word in ('piss', 'pissed', 'slave', 'slaves', 'porn', 'condom', 'sex', 'shit', 'shits', 'retarded', 'vagina', 'cunt', 'penis', 'ass', 'gays', 'gay', 'black', 'sex'):
                #      # print(word)
                #     innaprop_flag = 1

                # print(innaprop_flag)
                # if innaprop_flag == 0:
                joke_str = f"JOKE:{row[1]}{self.end_of_text_token}"
                self.joke_list.append(joke_str)
        # print(len(self.joke_list))

    def __len__(self):
        return len(self.joke_list)

    def __getitem__(self, item):
        return self.joke_list[item]


# Create Dataset object
dataset = JokesDataset()

# Utilize pytorch DataLoader to combine dataset object with different samplers
# Samplers are different strategies for providing data to models

joke_loader = DataLoader(dataset, batch_size=1, shuffle=True)

#################### Train Model #############################
# Train the model and save the model weights after each epoch
# Generate jokes with each version of weight to see what is best

model = model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
# scheduler = WarmupLinearSchedule(
#    optimizer, warmup_steps=WARMUP_STEPS, t_total=-1)
# TODO: Investigate what this is and proper number for TRAINING_STEPS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=TRAINING_STEPS,  last_epoch=-1)


proc_seq_count = 0
sum_loss = 0.0
batch_count = 0

tmp_jokes_tens = None
models_folder = "trained_models"
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

for epoch in range(EPOCHS):

    print(f"EPOCH {epoch} started" + '=' * 30)

    for idx, joke in enumerate(joke_loader):

        #################### "Fit as many joke sequences into MAX_SEQ_LEN sequence as possible" logic start ####
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
        loss.backward()
        sum_loss = sum_loss + loss.detach().data

        proc_seq_count = proc_seq_count + 1
        if proc_seq_count == BATCH_SIZE:
            proc_seq_count = 0
            batch_count += 1
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()

        if batch_count == 100:
            print(f"sum loss {sum_loss}")
            batch_count = 0
            sum_loss = 0.0

    # Store the model after each epoch to compare the performance of them
    torch.save(model.state_dict(), os.path.join(
        models_folder, f"gpt2_joker_{epoch}.pt"))
