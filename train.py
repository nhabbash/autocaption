import time
import numpy as np
import torch
from torch import nn
from models import Encoder, Decoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from data import Vocabulary, get_loaders

def train_models():

    epochs_since_improvement = 0 # Early stop
    best_bleu4 = 0

    # Logs
    train_losses = []
    val_losses = []
    val_bleu = []

    vocab = Vocabulary()

    cfg = { 
          'device' : "cuda" if torch.cuda.is_available() else "cpu",
          'train_batch_size' : 32,
          'test_batch_size' : 1000,
          'n_epochs' : 100,
          'seed' : 0,
          'log_interval' : 5,
          'save_model' : True,
          'lr' : 4e-4,
          "momentum": 0.01,
          'hidden_dim': 512,
          'emb_dim': 256,
          'n_layers': 1,
          'dropout': 0.5,
          'optimizer': None,
          }

    # Models
    encoder = Encoder(cfg["emb_dim"], cfg["momentum"]).to(cfg["device"])
    decoder = Decoder(vocab.max_len, len(vocab), cfg["emb_dim"], cfg["n_layers"], cfg["hidden_dim"], cfg["dropout"]).to(cfg["device"])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = (list(filter(lambda p: p.requires_grad, encoder.parameters()))+
            list(filter(lambda p: p.requires_grad, decoder.parameters())))
    optimizer = torch.optim.Adam(params, lr=cfg["lr"])
    cfg["optimizer"] = optimizer

    # Get loaders
    train_loader, val_loader, test_loader = get_loaders(cfg["train_batch_size"], 0)

    for epoch in range(cfg["n_epochs"]):
        if epochs_since_improvement == 20:
            break
        
        train_loss = train(loader=train_loader,
                encoder=encoder,
                decoder=decoder,
                criterion=criterion,
                opt=optimizer,
                epoch = epoch,
                cfg = cfg)

        
        val_loss, bleu4 = validate(loader=val_loader,
                encoder=encoder,
                decoder=decoder,
                criterion=criterion,
                epoch = epoch,
                cfg = cfg)
        
        # History
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_bleu.append(bleu4)
        
        if bleu4 > best_bleu4:
            best_bleu4 = bleu4
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

    if cfg["save_model"]:
        torch.save(encoder.state_dict(), "data/models/encoder.ckpt")
        torch.save(decoder.state_dict(), "data/models/decoder.ckpt")

    return [train_losses, val_losses, val_bleu]

def train(loader, encoder, decoder, criterion, opt, epoch, cfg):

    start = time.time()
    total_steps = len(loader)
    for i, (imgs, caps, lengths, _) in enumerate(loader):
        
        imgs = imgs.to(cfg["device"])
        caps = caps.to(cfg["device"])
        lengths = lengths.to(cfg["device"])
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
        
        if i % cfg["log_interval"] == 0:
            print('TRAIN: Epoch [{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, i, total_steps, loss.item(), np.exp(loss.item()))) 
    return loss.item()
            
def validate(loader, encoder, decoder, criterion, epoch, cfg):

    start = time.time()
    total_steps = len(loader)

    bleu_batch = []
    with torch.no_grad():
        for i, (imgs, caps, lengths, all_caps) in enumerate(loader):
            
            imgs = imgs.to(cfg["device"])
            caps = caps.to(cfg["device"])
            lengths = lengths.to(cfg["device"])
            targets = pack_padded_sequence(caps, lengths, batch_first=True)[0]
            
            # Forward+backward
            features = encoder(imgs)
            outputs = decoder(features, caps, lengths)
            packed_outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
            loss = criterion(packed_outputs, targets)

            # TODO: find a way to unpack output of decoder to extract predictions
            _, predicted = outputs.max(2)
            candidates = predicted.to("cpu").numpy()
            bleu_score = bleu_eval(candidates, all_caps)

            bleu_batch.append(bleu_score)

            if i % cfg["log_interval"] == 0:
                print('VAL: Epoch [{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}, BLEU4: {:5.4f}'.format(epoch, i, total_steps, loss.item(), np.exp(loss.item()), bleu_score))

    mean_bleu = sum(bleu_batch)/len(bleu_batch)
    print("Mean BLEU4: {:5.4f}".format(mean_bleu))
    return loss.item(), mean_bleu

def bleu_eval(candidates, all_caps):
    references = []
    all_caps = all_caps[0]

    for cap in all_caps:
        c = list(map(lambda c: [w for w in c if w not in {0, 1, 2}], cap)) # Remove <start>, <end> and <pad>
        references.append(c)

    score = corpus_bleu(references, candidates, smoothing_function=SmoothingFunction().method1)
    return score