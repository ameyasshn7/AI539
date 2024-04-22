from collections import Counter 
from re import sub, compile
import matplotlib.pyplot as plt
import numpy as np
import string
from datasets import load_dataset

class UnimplementedFunctionError(Exception):
	pass


class Vocabulary:

	def __init__(self, corpus, min_freq=50):

		self.min_freq = min_freq
		self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
		self.size = len(self.word2idx)

	def most_common(self, k):
		freq = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
		return [t for t,f in freq[:k]]


	def text2idx(self, text):
		tokens = self.tokenize(text)
		return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['UNK'] for t in tokens]

	def idx2text(self, idxs):
		return [self.idx2word[i] if i in self.idx2word.keys() else 'UNK' for i in idxs]


	###########################
	## TASK 1.1           	 ##
	###########################
	def tokenize(self, text):
		new_text = text.translate(str.maketrans('', '', string.punctuation))
		tokens = new_text.lower().split()
		return tokens

	###########################
	## TASK 1.2            	 ##
	###########################
	def build_vocab(self, corpus):
		self.freq = Counter()
		self.word2idx = {}
		self.idx2word = {}
		for s in corpus:
			tokens = self.tokenize(s)
			self.freq.update(tokens)

		idx = 1
		for word, c in self.freq.items():
			if c >= self.min_freq:  
				self.word2idx[word] = idx
				self.idx2word[idx] = word
				idx += 1

		self.word2idx['UNK'] = 0
		self.idx2word[0] = 'UNK'
		self.freq = dict(self.freq)
		return self.word2idx, self.idx2word, self.freq


	###########################
	## TASK 1.3              ##
	###########################
	def make_vocab_charts(self):
		freq_sorted = sorted(self.freq.values(), reverse=True)
		# tokens = list(self.freq.keys())
		# tokens_sorted = sorted(tokens, key=lambda x: self.freq[x], reverse=True)

		plt.figure(figsize=(10, 5))
		plt.subplot(1, 2, 1)
		plt.plot(freq_sorted)
		plt.title('Token Frequency Distribution')
		plt.xlabel('Token ID(Sorted by frequency)')
		plt.ylabel('Frequency')
		plt.yscale('log')  

		total = sum(freq_sorted)
		cumulative_sum = np.cumsum(freq_sorted)
		cumulative_fraction = cumulative_sum / total

		plt.subplot(1, 2, 2)
		plt.plot(cumulative_fraction)
		plt.title('Cumulative Fraction Covered')
		plt.xlabel('Token ID (sorted by frequency)')
		plt.ylabel('Fraction of Token Occurences Covered')

		plt.tight_layout()
		plt.show()




