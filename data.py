import pandas as pd
import numpy as np
import os.path
import string
import json
from collections import defaultdict, Counter
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, sampler
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

class Vocabulary(object):
    def __init__(self,
                 rebuild_vocab=False,
                 vocab_file="./data/vocab.json",
                 captions_file="./data/captions.json",
                 freq_threshold=5):

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
                for _, captions in collection.items():
                    for caption in captions["captions"]:
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


class CaptionDataset(Dataset):

    def __init__(self, split, precomp_features=False, transform=None):
        """
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param precomp_features: whether to use precomputed features or full images
        :param transform: image transform pipeline
        """
        self.vocab = Vocabulary()
        self.transform = transform
        self.precomp_features = precomp_features

        self.split = split
        split_dataset = {
            "TRAIN": "data/captions/Flickr_8k.trainImages.txt",
            "VAL":   "data/captions/Flickr_8k.devImages.txt", 
            "TEST":  "data/captions/Flickr_8k.testImages.txt"
        }
        assert self.split in split_dataset.keys()

        # Get paths and image ids
        self.paths = pd.read_csv(split_dataset[self.split], sep="\n", squeeze=True)
        self.paths.dropna()
        self.ids = [x.split('.')[0] for x in self.paths]

        # Use precomputed features if asked to
        if self.precomp_features:
            features = load(open("data/features.pkl", 'rb'))
            self.features = {k: features[k] for k in self.frame}
        else:
            pass
        
        # Get captions and lengths
        with open("data/captions.json", "r", encoding='utf-8') as f:
            captions = json.load(f)
            self.captions = {k: v["captions"] for k, v in captions.items() if k in self.ids}

            self.caption_lengths = defaultdict(list)
            for k, v in captions.items():
                # To consider every caption for each image
                #for i, length in enumerate(v["lengths"]):
                #    self.caption_lengths[length].append((k, i))
                self.caption_lengths[v["lengths"][0]].append(k)

    def get_raw_item(self, idx):
        path = "data/images/" + self.paths[idx]
        image = Image.open(path).convert("RGB")
        captions = self.captions[self.ids[idx]]

        return image, captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):

        if self.precomp_features:
            pass
            # TODO Using precomputed features
        else:
            path = "data/images/" + self.paths[idx]
            image = Image.open(path).convert("RGB")

            if self.split == "TRAIN":
                # Return image and caption
                if self.transform is not None:
                    image = self.transform(image) 

                # Encoding caption
                tokens = self.captions[self.ids[idx]][0].split()
                encoded = []
                encoded.append(self.vocab(self.vocab.start_word))
                encoded.extend([self.vocab(token) for token in tokens])
                encoded.append(self.vocab(self.vocab.end_word))
                encoded = torch.Tensor(encoded).long()

                return image, encoded

            elif self.split =="VAL": 
                # Return image and all its caption for BLEU scoring
                if self.transform is not None:
                    image = self.transform(image) 

                # Encoding caption
                captions = self.captions[self.ids[idx]]

                tokens = self.captions[self.ids[idx]][0].split()
                encoded = []
                encoded.append(self.vocab(self.vocab.start_word))
                encoded.extend([self.vocab(token) for token in tokens])
                encoded.append(self.vocab(self.vocab.end_word))
                encoded = torch.Tensor(encoded).long()

                encoded_captions = []
                for cap in captions:
                    cap = cap.split()
                    en = []
                    en.append(self.vocab(self.vocab.start_word))
                    en.extend([self.vocab(token) for token in cap])
                    en.append(self.vocab(self.vocab.end_word))
                    # Padding to max length for batching TODO: find max length of all captions
                    en.extend([self.vocab(self.vocab.pad_word) for i in range(50-len(en))])
                    encoded_captions.append(en)
                encoded_captions = torch.Tensor(encoded_captions).long()

                return image, encoded, encoded_captions

            else:
                 # Return only image for the TEST split
                orig_image = np.array(image)
                if self.transform is not None:
                    image = self.transform(image) 

            return orig_image, image
    
    def get_indices(self):
        lengths_prob = [k for k, v in self.caption_lengths.items() for repetitions in range(len(v))]
        sel_length = np.random.choice(lengths_prob)
        ids = self.caption_lengths[sel_length]

        indices = [self.ids.index(e) for e in ids if e in self.ids]
        return indices

def get_loader(split, batch_size, n_workers=0):
    transform_train = transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(), 
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                            (0.229, 0.224, 0.225))])

    transform_val = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(), 
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                            (0.229, 0.224, 0.225))])

    if split == "TRAIN":
        dataset = CaptionDataset(split="TRAIN", transform=transform_train)
        # Random sample of a caption length
        indices = dataset.get_indices()
        initial_sampler = sampler.SubsetRandomSampler(indices=indices)
        batch_sampler = sampler.BatchSampler(sampler=initial_sampler, batch_size=batch_size, drop_last=False)
        loader = torch.utils.data.DataLoader(dataset,
                                            num_workers=n_workers,
                                            batch_sampler=batch_sampler)

    elif split == "VAL":                                        
        dataset = CaptionDataset(split="VAL", transform=transform_val)
         # Random sample of a caption length
        indices = dataset.get_indices()
        initial_sampler = sampler.SubsetRandomSampler(indices=indices)
        batch_sampler = sampler.BatchSampler(sampler=initial_sampler, batch_size=batch_size, drop_last=False)
       
        loader = torch.utils.data.DataLoader(dataset,
                                            num_workers=n_workers,
                                            batch_sampler=batch_sampler)
    elif split == "TEST":                                
        dataset = CaptionDataset(split="TEST",transform=transform_val)
        indices = dataset.get_indices()
        initial_sampler = sampler.SubsetRandomSampler(indices=indices)
        batch_sampler = sampler.BatchSampler(sampler=initial_sampler, batch_size=batch_size, drop_last=False)
        loader = torch.utils.data.DataLoader(dataset,
                                            num_workers=n_workers,
                                            batch_sampler=batch_sampler)

    return loader