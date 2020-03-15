from torchvision import transforms
from models import Encoder
from PIL import Image
from pickle import dump
from collections import defaultdict, Counter
from data import Vocabulary
import string
import json
import os

import urllib.request
from zipfile import ZipFile 

def download_dataset():

    print("Downloading Flickr8k Images Dataset")
    url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    filename, _ = urllib.request.urlretrieve(url)
    
    with ZipFile(filename, 'r') as zip:
        files = [n for n in zip.namelist() 
                   if n.startswith('Flicker8k_Dataset/')]
        print("Extracting files in ./data/images")
        zip.extractall(path="./data/", members = files)

    os.rename("./data/Flicker8k_Dataset/", "./data/images")

def extract_features(dir="data/images"):
    enc = Encoder()
    features = defaultdict()

    compose = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    for filename in os.listdir(dir):
        path = dir + "/" + filename
        image = compose(Image.open(path).convert("RGB"))
        feature = enc(image.unsqueeze(0)).squeeze(0)
        
        image_id = filename.split(".")[0]
        features[image_id] = feature
        
        print(">{}".format(filename))
    
    print("Extracted features: {}".format(len(features)))

    # Saving to file
    print("Saving...")
    dump(features, open("data/features.pkl", "wb"))
        
def preprocess_captions(file="data/captions/Flickr8k.token.txt"):
    f = open(file, "r")
    txt = f.read()
    f.close()

    # Read captions
    collection = defaultdict(lambda: defaultdict(list))

    for line in txt.split("\n"):
        tokens = line.split()
        
        if len(tokens) < 2:
            continue

        image_id, caption = tokens[0], tokens[1:]
        image_id = image_id.split(".")[0]
        caption = " ".join(caption)
        
        collection[image_id]["captions"].append(caption)

    # Preprocess captions
    for key, captions in collection.items():
        for idx, c in enumerate(captions["captions"]):

            # Lowercasing, punctuation removal, tokenization
            c = c.lower()
            c = c.translate(str.maketrans("", "", string.punctuation))
            c = c.split()
           
            # (Short) stopword removal (only 1-char words and words with numbers)
            c = [word for word in c if len(word)>1]
            c = [word for word in c if word.isalpha()]

            captions["captions"][idx] = " ".join(c)
            captions["lengths"].append(len(c))

    # Saving captions and vocab to file
    with open("data/captions.json", "w", encoding='utf-8') as f:
        json.dump(collection, f)

def setup_data():
    extract_features()
    preprocess_captions()

def decode_cap(encoded_cap, v):
    return " ".join([v.idx2word[idx] for idx in encoded_cap if idx not in [v(v.start_word), v(v.end_word), v(v.pad_word)]])

def get_caption(img, encoder, decoder, vocab):
    # Generate a caption from the image
    features = encoder(img).unsqueeze(1)
    sample = decoder.sample(features)
    # Convert word_ids to words
    sentence = decode_cap(sample, vocab)
    return sentence