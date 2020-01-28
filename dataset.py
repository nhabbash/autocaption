import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
import json
from PIL import Image
from collections import defaultdict, Counter

class Vocabulary(object):

    def __init__(self):
        self.words = defaultdict()
        self.index = defaultdict()
        self.word_freq = Counter()
        self.idx = 0

    def add(self, word):
        if not word in self.words:
            self.words[word] = self.idx
            self.index[self.idx] = word
            self.idx += 1
    
    def __call__(self, word):
        if not word in self.words:
            return self.words["<unk>"]
        return self.words[word]

    def __len__(self):
        return len(self.words)

    def save(self):
        with open("data/vocab.json", "w") as f:
            json.dump([self.words]+[self.index], f)

    def load(self):
        with open("data/vocab.json", "r") as f:
            v = json.load(f)
            self.words = v[0]
            self.index = v[1]


class CaptionDataset(Dataset):

    def __init__(self, split, precomp_features=False, transform=None):
        """
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param precomp_features: whether to use precomputed features or full images
        :param transform: image transform pipeline
        """
        self.transform = transform
        self.precomp_features = precomp_features

        self.split = split
        split_dataset = {
            "TRAIN": "data/text/Flickr_8k.trainImages.txt",
            "VAL":   "data/text/Flickr_8k.devImages.txt", 
            "TEST":  "data/text/Flickr_8k.testImages.txt"
        }
        assert self.split in split_dataset.keys()

        # Get paths and image ids
        self.paths = pd.read_csv(split_dataset[self.split], sep="\n", squeeze=True)
        self.paths.dropna()
        self.ids = pd.Series([x.split('.')[0] for x in self.paths])

        # Features
        if self.precomp_features:
            features = load(open("data/features.pkl", 'rb'))
            self.features = {k: features[k] for k in self.frame}
        else:
            pass
        
        # Captions
        with open("data/captions.json", "r") as f:
            self.captions = json.load(f)
            self.captions = {k: v for k, v in self.captions.items() if k in self.ids.values}

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        if self.precomp_features:
            # Feature
            img_id = self.frame[idx]
            feature = self.features[img_id]
        else:
            path = "data/images/" + self.paths[idx]
            image = Image.open(path).convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        # Caption
        target = self.captions[self.ids[idx]]["encoded"][0]
        target = torch.Tensor(target)
        if self.split is "TRAIN":
            return image, target
        else:
            all_captions = torch.LongTesnor(self.captions[self.ids[idx]]["encoded"])
            return image, target, all_captions
       

def collate_fn(data):
    """
    <Author: Yunjey Choi, https://github.com/yunjey>
    Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    print(captions)
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths
