import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torchtext.datasets import UDPOS
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchtext
import portalocker


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data, test_data, val_data = UDPOS()

tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for _, tags, text in data_iter:
        if isinstance(text, list):
            # If the text is already a list of tokens, yield them directly
            yield text
        else:
            # Otherwise, apply the tokenizer
            yield tokenizer(text)
token_vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<unk>"])
token_vocab.set_default_index(token_vocab["<unk>"])

def yield_tags(data_iter):
    for _, tags, _ in data_iter:
        yield tags

# Build a vocabulary of POS tags
tag_vocab = build_vocab_from_iterator(yield_tags(train_data), specials=["<unk>"])
tag_vocab.set_default_index(tag_vocab["<unk>"])
