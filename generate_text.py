from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import torch
import logging
logging.getLogger().setLevel(logging.CRITICAL)


device = 'cpu'


# if torch.cuda.is_available():
#     device = 'cuda'
pt_model = 'gpt2'
print("Importing" + pt_model)

tokenizer = GPT2Tokenizer.from_pretrained(pt_model)
model = GPT2LMHeadModel.from_pretrained(pt_model)
model = model.to(device)
print(pt_model+"model imported")

# Function to first select topN tokens from the probability list and then based on the selected N word distribution
# get random token ID


def choose_from_top(probs, n=1):
    print("Selecting Word")
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob)  # Normalize
    choice = np.random.choice(n, 1, p=top_prob)
    token_id = ind[choice][0]
    return int(token_id)


def generate_some_text(input_str, text_len=50):
    print("Startong Word Generation")
    cur_ids = torch.tensor(tokenizer.encode(input_str)
                           ).unsqueeze(0).long().to(device)

    model.eval()
    with torch.no_grad():

        for i in range(text_len):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            # Take the first(only one) batch and the last predicted embedding
            softmax_logits = torch.softmax(logits[0, -1], dim=0)
            # Randomly(from the given probability distribution) choose the next word from the top n words
            next_token_id = choose_from_top(
                softmax_logits.to('cpu').numpy(), n=10)
            cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(
                device) * next_token_id], dim=1)  # Add the last word

        output_list = list(cur_ids.squeeze().to('cpu').numpy())
        output_text = tokenizer.decode(output_list)
        print(output_text)


# Tests
# Matrix Description
generate_some_text("The Matrix is everywhere. It is all around us. Even now, in this very room. You can see it when you look out your window or when you turn on your television. You can feel it when you go to work... when you go to church... when you pay your taxes. It is the world that has been pulled over your eyes to blind you from the truth.")
