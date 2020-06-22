from torchvision import transforms
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
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
        
def preprocess_captions(file="data/captions/Flickr8k.token.txt"):
    f = open(file, "r")
    txt = f.read()
    f.close()

    # Read captions
    collection = []

    for line in txt.split("\n"):
        tokens = line.split()
        
        if len(tokens) < 2:
            continue

        image_id, caption = tokens[0], tokens[1:]
        image_id = image_id.split(".")[0]
        caption = " ".join(caption)

        # Lowercasing, punctuation removal, tokenization
        caption = caption.lower()
        caption = caption.translate(str.maketrans("", "", string.punctuation))
        caption = caption.split()
        
        # (Short) stopword removal (only 1-char words and words with numbers)
        caption = [word for word in caption if len(word)>1]
        caption = [word for word in caption if word.isalpha()]
        
        if len(caption) > 1:
            caption = " ".join(caption)
            collection.append((caption, image_id))

    # Saving captions and vocab to file
    with open("data/captions.json", "w", encoding='utf-8') as f:
        json.dump(collection, f)

def setup_data():
    extract_features()
    preprocess_captions()

def decode_cap(encoded_cap, v):
    return " ".join([v.idx2word[idx] for idx in encoded_cap if idx not in [v(v.start_word), v(v.end_word), v(v.pad_word)]])

def get_caption(img, encoder, decoder, vocab, sample="beam", decode=True):
    # Generate a caption from the image
    features = encoder(img).unsqueeze(1)

    if sample == "beam":
        sample = decoder.sample_beam_search(features)
    else:
        sample = [decoder.sample(features)]

    if decode:
        # Convert word_ids to words
        sentences = [decode_cap(s, vocab) for s in sample]
        return sentences
    else:
        return sample

def bleu_eval(candidates, all_caps, v):
    references = []
    for batch in all_caps:
        batch_ref = []
        for cap in batch:
            if isinstance(cap, str):
                cap = []
            else:
                cap = [w.item() for w in cap if w not in [v(v.start_word), v(v.end_word), v(v.pad_word)]] # Strip <start>, <end> and <pad>
            batch_ref.append(cap)
        references.append(batch_ref)
    batch_bleu = corpus_bleu(references, candidates, smoothing_function=SmoothingFunction().method1)

    return batch_bleu

def early_stopping(val_bleus, patience=3):

    if patience > len(val_bleus):
        return False
    last_bleus = val_bleus[-patience:]
    best_bleu = max(last_bleus)

    #if len(set(last_bleus)) == 1:
    if np.std(last_bleus) < 0.01:
        return True

    if best_bleu in last_bleus:
        if best_bleu not in val_bleus[:len(val_bleus)-patience]:
            return False
        else:
            return True

    return True

