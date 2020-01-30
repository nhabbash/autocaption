import time
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from nltk.translate.bleu_score import corpus_bleu

def train(loader, encoder, decoder, criterion, opt, print_freq, epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start = time.time()
    total_steps = len(loader)
    for i, (imgs, caps, lengths, _) in enumerate(loader):
        
        imgs = imgs.to(device)
        caps = caps.to(device)
        lengths = lengths.to(device)
        targets = pack_padded_sequence(caps, lengths, batch_first=True)[0]
      
        # Forward+backward
        features = encoder(imgs)
        outputs = decoder(features, caps, lengths)
        packed_outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
        loss = criterion(packed_outputs, targets)
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        opt.step()
        
        if i % print_freq == 0:
            print('TRAIN: Epoch [{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, i, total_steps, loss.item(), np.exp(loss.item()))) 
    return loss.item()
            
def validate(loader, encoder, decoder, criterion, print_freq, epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start = time.time()
    total_steps = len(loader)
    with torch.no_grad():
        for i, (imgs, caps, lengths, all_caps) in enumerate(loader):
            
            imgs = imgs.to(device)
            caps = caps.to(device)
            lengths = lengths.to(device)
            targets = pack_padded_sequence(caps, lengths, batch_first=True)[0]
            
            # Forward+backward
            features = encoder(imgs)
            outputs = decoder(features, caps, lengths)
            packed_outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
            loss = criterion(packed_outputs, targets)

            # TODO: find a way to unpack output of decoder to extract predictions
            _, predicted = outputs.max(2)
            candidates = predicted.to("cpu").numpy()
            bleu_score = bleu_eval(outputs, all_caps)
            
            if i % print_freq == 0:
                print('VAL: Epoch [{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}, BLEU4: {:5.4f}'.format(epoch, i, total_steps, loss.item(), np.exp(loss.item()), bleu_score)) 
    return loss.item(), bleu_score

def bleu_eval(candidates, all_caps):
    references = []
    all_caps = all_caps[0]
    for cap in all_caps:
        c = list(map(lambda c: [w for w in c if w not in {0, 1, 2}], cap)) # Remove <start>, <end> and <pad>
        references.append(c)

    score = corpus_bleu(references, candidates)
    return score