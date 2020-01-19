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
TRAINING_STEPS = 5000
MAX_SEQ_LEN = 400

device = 'cpu'
# if torch.cuda.is_available():
#     device = 'cuda'
#     print("GPU detected, utilizing GPU")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob)  # Normalize
    choice = np.random.choice(n, 1, p=top_prob)
    token_id = ind[choice][0]
    return int(token_id)


###### Test Model #######
MODEL_EPOCH = 4

models_folder = "trained_models"

model_path = os.path.join(models_folder, f"gpt2_joker_{MODEL_EPOCH}.pt")
model.load_state_dict(torch.load(model_path))

jokes_output_file_path = f'joke_gen_output/generated_{MODEL_EPOCH}.jokes'

model.eval()

# If generated jokes list already exists, replace it
if os.path.exists(jokes_output_file_path):
    os.remove(jokes_output_file_path)

joke_num = 0
with torch.no_grad():

    for joke_idx in range(1000):

        joke_finished = False

        cur_ids = torch.tensor(tokenizer.encode("JOKE:")
                               ).unsqueeze(0).to(device)

        for i in range(100):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            # Take the first(from only one in this case) batch and the last predicted embedding
            softmax_logits = torch.softmax(logits[0, -1], dim=0)
            if i < 3:
                n = 20
            else:
                n = 3
            # Randomly(from the topN probability distribution) select the next word
            next_token_id = choose_from_top(
                softmax_logits.to('cpu').numpy(), n=n)
            cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(
                device) * next_token_id], dim=1)  # Add the last word to the running sequence

            if next_token_id in tokenizer.encode('<|endoftext|>'):
                joke_finished = True
                break

        if joke_finished:

            joke_num = joke_num + 1

            output_list = list(cur_ids.squeeze().to('cpu').numpy())
            output_text = tokenizer.decode(output_list)

            with open(jokes_output_file_path, 'a') as f:
                f.write(f"{output_text} \n\n")
