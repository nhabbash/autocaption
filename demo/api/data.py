import os.path
import string
import json
from collections import defaultdict, Counter

class Vocabulary(object):
    def __init__(self,
                 freq_threshold,
                 rebuild_vocab=False,
                 vocab_file="./data/vocab.json",
                 captions_file="./data/captions.json",
                 ):
        self.rebuild_vocab = rebuild_vocab
        self.vocab_file = vocab_file
        self.captions_file = captions_file
        self.freq_threshold = freq_threshold

        self.pad_word = "<pad>"
        self.start_word = "<start>"
        self.end_word = "<end>"
        self.unk_word = "<unk>"

        self.word2idx = defaultdict()
        self.idx2word = list()
        self.idx = 0
        self.words_freq = Counter()
        self.get_vocab()

    def get_vocab(self):
        if os.path.exists(self.vocab_file) and not self.rebuild_vocab:
            with open(self.vocab_file, "r") as f:
                v = json.load(f)
                self.word2idx = v["word2idx"]
                self.idx2word = v["idx2word"]
                self.words_freq = Counter(v["words_freq"])
                self.idx = len(self.idx2word)+1
        else:
            self.build_vocab()
            with open(self.vocab_file, "w", encoding='utf-8') as f:
                json.dump(
                    {"word2idx" : self.word2idx,
                     "idx2word" : self.idx2word,
                     "words_freq" : self.words_freq}, f)
        
    def build_vocab(self):
        if os.path.exists(self.captions_file):
            with open(self.captions_file, "r") as f:
                collection = json.load(f)
                for caption, _ in collection:
                    self.words_freq.update(caption.split())

            # Creating vocabulary
            words = {w for w in self.words_freq.keys() if self.words_freq[w] > self.freq_threshold}
            self.add(self.pad_word)   #0
            self.add(self.unk_word)   #1
            self.add(self.start_word) #2
            self.add(self.end_word)   #3

            for w in words:
                self.add(w)

    def add(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word.append(word)
            self.idx += 1
    
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
