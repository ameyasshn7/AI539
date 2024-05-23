######################################################
# Use these package versions
#!pip install torchtext==0.6.0 torch==1.13.1
######################################################


import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] =':16:8' #This is a command to reduce non-deterministic behavior in CUDA
import warnings
warnings.simplefilter("ignore", UserWarning)
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import get_tokenizer
import sys
import argparse
from queue import PriorityQueue
from LanguageModel import LanguageModel
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')



def main():
  chkpt = "OSU-NLP-HW3-Skeleton-2/OSU-NLP-HW3-Skeleton/got_language_model"

  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logging.info('Using device: {}'.format(dev))

  logging.info("Loading tokenizer and vocab from vocab.pkl")  
  text_field = pickle.load(open("OSU-NLP-HW3-Skeleton-2/OSU-NLP-HW3-Skeleton/vocab.pkl", "rb"))
  vocab_size = len(text_field.vocab.itos)

  logging.info("Loading checkpoint {}".format(chkpt))
  lm = LanguageModel(vocab_size).to(dev)
  lm.load_state_dict(torch.load(chkpt,map_location='cpu'))
  lm.eval()


  p = "the night is dark and full of terrors"

  # Torch is a bit frustrating at times and some things that ought to be deterministic are not.
  # This is an attempt to resolve that, but it doesn't work 100% of the time
  torch.use_deterministic_algorithms(True)
  seed = 42
  mlen = 150

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Vanilla Sampling -----------")
  print(sample(lm, text_field, prompt=p, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n------- Temp-Scaled Sampling 0.0001 -------")
  print(sample(lm, text_field, prompt=p, temp=0.0001, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n------- Temp-Scaled Sampling 100 --------")
  print(sample(lm, text_field, prompt=p, temp=100, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-k Sampling 1 -----------")
  print(sample(lm, text_field, prompt=p, k=1, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-k Sampling 20 -----------")
  print(sample(lm, text_field, prompt=p, k=20, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-p Sampling 0.001 -----------")
  print(sample(lm, text_field, prompt=p, p=0.001, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-p Sampling 0.75 -----------")
  print(sample(lm, text_field, prompt=p, p=0.75, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-p Sampling 1 -----------")
  print(sample(lm, text_field, prompt=p, p=1, max_len=mlen))


  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=1 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=1, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=10 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=10, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=50 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=50, max_len=mlen))

  print()

############################################################################################
# TASK 2.1
############################################################################################

def beamsearch(model, text_field, beams=5, prompt="", max_len=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    
    prompt_tokens = text_field.process([text_field.tokenize(prompt.lower())])
    prompt_tokens = prompt_tokens.squeeze(0).to(device)
    initial_st = (torch.zeros(model.rnn.num_layers, 1, model.hidden_size).to(device),
                     torch.zeros(model.rnn.num_layers, 1, model.hidden_size).to(device))
    beam_list = [(prompt_tokens, 0, *initial_st)]

    for _ in range(max_len - len(prompt_tokens)):
        new_beam_list = []
        for beam in beam_list:
            seq, probs, h, c = beam

            logits, h, c = model(seq[-1].unsqueeze(0), h, c)
            logits = logits.squeeze(0)


            top_probs, top_indices = torch.topk(logits, beams)
            top_probs = top_probs.detach()
            top_indices = top_indices.detach()
            top_indices = top_indices.view(-1)  

            for i in range(beams):
                if top_probs.ndim > 1:
                    new_prob = top_probs[0, i].item()
                else:
                    new_prob = top_probs[i].item()


                new_index = top_indices[i].unsqueeze(0) 
                new_seq = torch.cat([seq, new_index.view(1,1)], dim=0)
                new_beam_list.append((new_seq, probs + new_prob, h, c))

        
        beam_list = sorted(new_beam_list, key=lambda x: x[1], reverse=True)[:beams]

    best_seq, _, _, _ = beam_list[0]
    decoded_string = reverseNumeralize(best_seq, text_field)
    return decoded_string

############################################################################################
# TASK 1.1
############################################################################################
def top_k_sampling(logits, top_k=0, filter_value=-float('Inf')):
    
    assert logits.dim() == 2

    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
    logits[indices_to_remove] = filter_value
    return logits

def top_p_sampling(logits, top_p=0.0, filter_value=-float('Inf')):

    if top_p == 0.0:
        return logits.fill_(filter_value)  

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices.argsort(dim=-1), sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value
    return logits


def sample(model, text_field, prompt="", max_len=50, temp=1.0, k=0, p=1):
  assert (k==0 or p==1), "Cannot combine top-k and top-p sampling"
  # decodedString = "Not implemented"

  tokenizer = get_tokenizer('basic_english')
  tokens = tokenizer(prompt)
  input_ids = [text_field.vocab.stoi[token] for token in tokens]

  input_tensor = torch.tensor(input_ids).unsqueeze(1).to('cpu')

  
  h = torch.zeros(model.rnn.num_layers, 1, model.hidden_size).to('cpu')
  c = torch.zeros(model.rnn.num_layers, 1, model.hidden_size).to('cpu')
  logits, h ,c =  model(input_tensor,h,c)
  outputs = input_ids.copy()


  for i in range(max_len):
        logits, h, c = model(input_tensor, h, c)
        logits = logits.squeeze(1)  

        if temp != 1.0:
            logits /= temp
        elif k > 0:
           logits = top_k_sampling(logits,k)
        elif p != 1:
           logits = top_p_sampling(logits,p)

        probabilities = F.softmax(logits, dim=-1)


        next_word_tensor = torch.multinomial(probabilities, 1)
        next_word_idx = next_word_tensor[0,0].item()


        outputs.append(next_word_idx)
        input_tensor = torch.tensor([[next_word_idx]], dtype=torch.long).to('cpu')

        if next_word_idx == text_field.vocab.stoi['<eos>']:
            break

  decoded_string = ' '.join(text_field.vocab.itos[idx] for idx in outputs)
  return decoded_string



############################################################################################

def reverseNumeralize(numeralized_string, text_field):
  strings = [text_field.vocab.itos[i] for i in numeralized_string]
  return " ".join(strings)

if __name__ == "__main__":
  main()
