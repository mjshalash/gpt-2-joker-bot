from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import torch
import logging
logging.getLogger().setLevel(logging.CRITICAL)


device = 'cpu'


if torch.cuda.is_available():
    device = 'cuda'
pt_model = 'gpt2'
print("Importing " + pt_model)

tokenizer = GPT2Tokenizer.from_pretrained(pt_model)
model = GPT2LMHeadModel.from_pretrained(pt_model)
model = model.to(device)
print(pt_model+"model imported")

# Function to first select topN tokens from the probability list and then based on the selected N word distribution
# get random token ID


def choose_from_top(probs, n=40):
    print("Selecting Word")
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob)  # Normalize
    choice = np.random.choice(n, 1, p=top_prob)
    token_id = ind[choice][0]
    return int(token_id)


def generate_some_text(input_str, text_len=500):
    print("Starting Word Generation")
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

        # Write to file
        outFile = open("text_gen_output/" + pt_model + ".txt", "a")
        outFile.write("\n\n----Prompt---- \n")
        outFile.write(input_str + "\n")
        outFile.write("----Generated Text---- \n")
        outFile.writelines(output_text)
        outFile.close()

        print("File Successfully Written")


# Tests
# Matrix Description
generate_some_text("Whoa! Yeah, Cleveland. What’s up, Cleveland? How you all feeling? Everybody good? Y’all good? Everybody straight? Cold as shit out this bitch, ain’t it? I don’t like that— All this snow. I don’t like that shit. Y’all got that slushy shit, that slipping snow. Hey, look— God damn it. There’s slush. There’s slush right there. Watch the slush. I don’t like that shit. Y’all ain’t supposed to have snow out here. I don’t like that shit. Y’all ain’t supposed to have snow out here. Good year for y’all though. Right now got— Before I even get started, shouts out to— — Hey, sugar foot. How you doing? How you doing, sweetie? Get it out now. Before we get started, shouts out to my man Shaq up front showing love. The cavs. My man LeBron in the house. Shout out to our boy LeBron in the house. It’s a good year for y’all. A good year for y’all. Y’all might do it. Y’all might do it. Y’all might do it. Y’all might do it. All right, y’all got me up here for a while, all right? And this time— About an hour. I’m gonna be up here about an hour. Now in this time y’all gonna hear a lot of stuff. I’m not gonna lie to y’all. Don’t judge me. I don’t want nobody judging me. I love to be honest when I’m on stage.")
