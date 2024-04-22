
from datasets import load_dataset
from Vocabulary import Vocabulary
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.utils.extmath import randomized_svd
import logging
import itertools
from sklearn.manifold import TSNE
import pandas as pd
from collections import defaultdict
import os
import random
import string
random.seed(42)
np.random.seed(42)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

class UnimplementedFunctionError(Exception):
	pass


###########################
## TASK 2.2              ##
###########################

def compute_cooccurrence_matrix(corpus, vocab):
	window_size = 2
	vocab_size = len(vocab.word2idx)
	C = np.zeros((vocab_size, vocab_size), dtype=int)
	word_2_indx = vocab.word2idx

	for s in corpus:
		words= vocab.tokenize(s)
		indx = [word_2_indx.get(word,word_2_indx['UNK']) for word in words]
		for center_indx, center_word in enumerate(indx):
			start = max(0,center_indx - vocab_size)
			end = min(len(indx), center_indx + vocab_size + 1)

			for context in range(start, end, window_size):
				if context != center_indx:
					context_word = indx[context]
					if context_word is not None:
						C[center_word,context_word] += 1

	return C



###########################
## TASK 2.3              ##
###########################

def compute_ppmi_matrix(corpus, vocab):
    C = compute_cooccurrence_matrix(corpus, vocab)

    total = np.sum(C)
    const = 1e-10  

    probs = C / (total + const)
    sum_rows = np.sum(probs, axis=1) + const
    sum_cols = np.sum(probs, axis=0) + const

    pmi = np.log((probs + const) / (sum_rows[:, None] * sum_cols[None, :]))
    pmi = np.nan_to_num(pmi, nan=0.0, posinf=0.0, neginf=0.0)

    PPMI = np.maximum(pmi, 0)
    return PPMI


	

################################################################################################
# Main Skeleton Code Driver
################################################################################################
def main_freq():

	logging.info("Loading dataset")
	dataset = load_dataset("ag_news")
	dataset_text =  [r['text'] for r in dataset['train']]
	dataset_labels = [r['label'] for r in dataset['train']]


	logging.info("Building vocabulary")
	vocab = Vocabulary(dataset_text)
	vocab.make_vocab_charts()
	plt.close()
	plt.pause(0.01)


	logging.info("Computing PPMI matrix")
	PPMI = compute_ppmi_matrix( [doc['text'] for doc in dataset['train']], vocab)


	logging.info("Performing Truncated SVD to reduce dimensionality")
	word_vectors = dim_reduce(PPMI)


	logging.info("Preparing T-SNE plot")
	plot_word_vectors_tsne(word_vectors, vocab)


def dim_reduce(PPMI, k=16):
	U, Sigma, VT = randomized_svd(PPMI, n_components=k, n_iter=10, random_state=42)
	SqrtSigma = np.sqrt(Sigma)[np.newaxis,:]

	U = U*SqrtSigma
	V = VT.T*SqrtSigma

	word_vectors = np.concatenate( (U, V), axis=1) 
	word_vectors = word_vectors / np.linalg.norm(word_vectors, axis=1)[:,np.newaxis]

	return word_vectors


def plot_word_vectors_tsne(word_vectors, vocab):
	coords = TSNE(metric="cosine", perplexity=50, random_state=42).fit_transform(word_vectors)

	plt.cla()
	top_word_idx = vocab.text2idx(" ".join(vocab.most_common(1000)))
	plt.plot(coords[top_word_idx,0], coords[top_word_idx,1], 'o', markerfacecolor='none', markeredgecolor='k', alpha=0.5, markersize=3)

	for i in tqdm(top_word_idx):
		plt.annotate(vocab.idx2text([i])[0],
			xy=(coords[i,0],coords[i,1]),
			xytext=(5, 2),
			textcoords='offset points',
			ha='right',
			va='bottom',
			fontsize=5)
	plt.show()


if __name__ == "__main__":
    main_freq()

