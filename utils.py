from torchvision import transforms
from models import Encoder
from PIL import Image
from pickle import dump
from collections import defaultdict, Counter
from data import Vocabulary
import string
import json
import os

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
        
def preprocess_captions(file="data/text/Flickr8k.token.txt", min_word_freq=50):
    f = open(file, "r")
    txt = f.read()
    f.close()

    # Read captions
    collection = defaultdict(lambda: defaultdict(list))
    vocab = Vocabulary()
    max_len = 0

    for line in txt.split("\n"):
        tokens = line.split()
        
        if len(tokens) < 2:
            continue

        image_id, caption = tokens[0], tokens[1:]
        image_id = image_id.split(".")[0]
        caption = " ".join(caption)
        
        collection[image_id]["raw"].append(caption)

    # Preprocess captions
    for key, captions in collection.items():
        for idx, c in enumerate(captions["raw"]):

            # Lowercasing, punctuation removal, tokenization
            c = c.lower()
            c = c.translate(str.maketrans("", "", string.punctuation))
            c = c.split()
           
            # (Short) stopword removal (only 1-char words and words with numbers)
            c = [word for word in c if len(word)>1]
            c = [word for word in c if word.isalpha()]
            vocab.word_freq.update(c)

            # Update max_len for padding
            if len(c) > max_len:
                max_len = len(c)

            captions["raw"][idx] = " ".join(c)

    # Creating vocabulary
    words = {w for w in vocab.word_freq.keys() if vocab.word_freq[w] > min_word_freq}
    vocab.add("<pad>") #0
    vocab.add("<start>") #1
    vocab.add("<end>") #2
    vocab.add("<unk>") #3
    vocab.max_len = max_len
    for w in words:
        vocab.add(w)

    # Encode captions
    for key, captions in collection.items():
        for idx, c in enumerate(captions["raw"]):
            
            encoded_c = [vocab("<start>")] +\
                        [vocab.words.get(word, vocab("<unk>")) for word in c.split()] +\
                        [vocab("<end>")] 
            captions["lengths"].append(len(encoded_c))
            encoded_c = encoded_c + [vocab("<pad>")] * (max_len - len(c.split()))
            captions["encoded"].append(encoded_c)
            

    # Saving captions and vocab to file
    with open("data/captions.json", "w", encoding='utf-8') as f:
        json.dump(collection, f)

    vocab.save()


def setup_data():
    extract_features()
    preprocess_captions()

def decode_cap(encoded_cap, vocab):
    return [vocab.indices[idx] for idx in encoded_cap.int().numpy()]