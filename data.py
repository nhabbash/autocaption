import pandas as pd
import numpy as np
import os.path
import string
import json
import torch
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.transforms as transforms
from collections import defaultdict, Counter
from PIL import Image

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


class CaptionDataset(Dataset):

    def __init__(self, split, batch_size, freq_threshold=5, transform=None):
        """
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param precomp_features: whether to use precomputed features or full images
        :param transform: image transform pipeline
        """
        self.vocab = Vocabulary(freq_threshold=freq_threshold)
        self.transform = transform
        self.batch_size = batch_size

        self.split = split
        split_dataset = {
            "TRAIN": "data/captions/Flickr_8k.trainImages.txt",
            "VAL":   "data/captions/Flickr_8k.devImages.txt", 
            "TEST":  "data/captions/Flickr_8k.testImages.txt",
            "COMPLETE": "",
        }
        assert self.split in split_dataset.keys()

        # Get paths and image ids
        if self.split!="COMPLETE":
            paths = pd.read_csv(split_dataset[self.split], sep="\n", squeeze=True)
            paths.dropna()
            self.ids = [x.split('.')[0] for x in paths]
        
        # Get captions and lengths
        with open("./data/captions.json", "r", encoding='utf-8') as f:
            self.captions = json.load(f)
            if self.split != "COMPLETE":
                self.captions = [element for element in self.captions if element[1] in self.ids]
            self.caption_lengths = [len(cap.split()) for cap, img_id in self.captions]

    def get_raw_item(self, idx):
        caption, image_id = self.captions[idx]
        path = "./data/images/" + image_id + ".jpg"
        image = Image.open(path).convert("RGB")

        return image, caption

    def __len__(self):
        if self.split == "TEST":
            return len(self.ids)
        return len(self.captions)

    def __getitem__(self, idx):
        
        caption = None
        if self.split == "TEST":
            image_id = self.ids[idx]
        else:
            caption, image_id = self.captions[idx]

        path = "./data/images/" + image_id + ".jpg"
        orig_image = Image.open(path).convert("RGB")
        
        # Return image and caption
        if self.transform is not None:
            image = self.transform(orig_image) 

        # Encoding caption
        if caption:
            tokens = caption.split()
            encoded = []
            encoded.append(self.vocab(self.vocab.start_word))
            encoded.extend([self.vocab(token) for token in tokens])
            encoded.append(self.vocab(self.vocab.end_word))
            encoded = torch.Tensor(encoded).long()

        if self.split in ["TRAIN", "COMPLETE"]:
            return image, encoded

        elif self.split in ["VAL", "TEST"]: 

            # Encoding all the image's caption
            captions = [cap for cap, i_id in self.captions if image_id == i_id]

            encoded_captions = []
            for cap in captions:
                cap = cap.split()
                en = []
                en.append(self.vocab(self.vocab.start_word))
                en.extend([self.vocab(token) for token in cap])
                en.append(self.vocab(self.vocab.end_word))

                # Padding to max length for batching
                en = en + [self.vocab(self.vocab.pad_word)] * (max(self.caption_lengths)+2-len(en))
                encoded_captions.append(en)
            encoded_captions = torch.Tensor(encoded_captions).long()

            if self.split == "TEST":
                return np.array(orig_image), image, captions
            else:
                return image, encoded, encoded_captions
    
    def get_indices(self, lenght=False):
        sel_length = np.random.choice(self.caption_lengths)
        indices = np.where(self.caption_lengths == sel_length)[0]
        indices = list(np.random.choice(indices, size=self.batch_size))
        if lenght:
            return indices, sel_length
        return indices

def get_loader(split, batch_size, n_workers=0, transform=0):
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
        dataset = CaptionDataset(split="TRAIN", transform=transform_train, batch_size=batch_size)
        indices = dataset.get_indices()
        init_sampler = sampler.SubsetRandomSampler(indices=indices)
        length_sampler = sampler.BatchSampler(sampler=init_sampler, batch_size=dataset.batch_size, drop_last=False)
        loader = DataLoader(dataset, batch_sampler=length_sampler, num_workers=n_workers)

    elif split == "VAL":                                        
        dataset = CaptionDataset(split="VAL", transform=transform_val, batch_size=batch_size)
        indices = dataset.get_indices()
        init_sampler = sampler.SubsetRandomSampler(indices=indices)
        length_sampler = sampler.BatchSampler(sampler=init_sampler, batch_size=dataset.batch_size, drop_last=False)
        loader = DataLoader(dataset, batch_sampler=length_sampler, num_workers=n_workers)

    elif split == "TEST":                                
        dataset = CaptionDataset(split="TEST",transform=transform_val, batch_size=batch_size)
        loader = DataLoader(dataset, batch_size=1, num_workers=n_workers)

    elif split == "COMPLETE":
        if transform == 0:
            transform = transform_train
        else:
            transform = transform_val
        dataset = CaptionDataset(split="COMPLETE", transform=transform, batch_size=batch_size)
        indices = dataset.get_indices()
        init_sampler = sampler.SubsetRandomSampler(indices=indices)
        length_sampler = sampler.BatchSampler(sampler=init_sampler, batch_size=dataset.batch_size, drop_last=False)
        loader = DataLoader(dataset, batch_sampler=length_sampler, num_workers=n_workers)

    return loader