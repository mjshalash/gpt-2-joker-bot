import sys
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

# TODO: Remove uneccessary variables
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


# TODO: Make model name a variable to automate organizing saved models
# Retrieve language model from huggingface AWS S3 Bucket

# Byte-pair encoder
# Transforms input text into recognizable input tokens
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# GPT2LMHeadModel is the GPT2Model with an extra linear layer
# Extra layer does inverse op of embedding layer to create dict. of possible word outcomes
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Important function
# Essentially it determines which word it chooses to output
# based on probabilities of possible words
# n parameter determines how many words/probabilities are considered


def choose_from_top(probs, n):
    # Partitions probability array, -n is axis to sort on
    # elements < -n left, elements >= -n to right
    # Then, take only right side of partition
    ind = np.argpartition(probs, -n)[-n:]

    # Normalize probabilities array
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob)

    # Choosing 1 random sample from prob array
    # USING given probabilities
    choice = np.random.choice(n, 1, p=top_prob)
    token_id = ind[choice][0]
    return int(token_id)


###### Model Selection #######
models_folder = "TM/VarTS/40K/"

model_path = os.path.join(models_folder, f"Trial2_2.pt")
model.load_state_dict(torch.load(model_path))

jokes_output_file_path = f'output/sample.jokes'

# Switch model to evalutation mode (self.training set to false)
# Some models behave differently when training vs testing
model.eval()

# If generated jokes list already exists, replace it
if os.path.exists(jokes_output_file_path):
    os.remove(jokes_output_file_path)

# Output joke number
joke_num = 0

# Parse Passed In Parameters
# N is first parameter for calling this program
n = int(sys.argv[1])


# Temporarily sets all requires_grad flags to false
# torch.Tensor has a requires_grad flag. When set to true, it tracks all ops on it
# When you finish, call backward() and compute gradients (backpropogation)
# no_grad() prevents tracking history and using memory when evaluating
# as we do not need to update our gradients
with torch.no_grad():

    # Output 10 Jokes
    for joke_idx in range(10):

        joke_finished = False

        # tensor = multidimensional array with variable num. of axis
        # Vector = 1-order tensor, Matrix = 2 order tensor, etc.
        # Creates tensor with joke data
        # Tokenizer.encode is encoding joke input
        # Finally, copy tensor to cpu
        cur_ids = torch.tensor(tokenizer.encode("JOKE:")
                               ).unsqueeze(0).to(device)

        for i in range(100):
            # print(f"{i}\n")
            # Process tensor through the model
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]

            # Take the first batch and the last predicted embedding
            # Use softmax to turn possible words into corresponding probabilities
            softmax_logits = torch.softmax(logits[0, -1], dim=0)

            # Randomly(from the topN probability distribution) select the next word
            # Pass in probabilities array to chooseFromTop function and select from top n
            # BASED on probabilities
            next_token_id = choose_from_top(
                softmax_logits.to('cpu').numpy(), n=n)

            # Concatenate chosen word_token to running sequence
            cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(
                device) * next_token_id], dim=1)

            # If machine decided to choose <|endoftext|> then we done
            if next_token_id in tokenizer.encode('<|endoftext|>'):
                joke_finished = True
                break

        # If the joke is finished, write joke to file
        if joke_finished:

            joke_num = joke_num + 1

            output_list = list(cur_ids.squeeze().to('cpu').numpy())
            output_text = tokenizer.decode(output_list)

            with open(jokes_output_file_path, 'a') as f:
                f.write(f"{joke_num} {output_text} \n\n")
