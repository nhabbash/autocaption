import time
import os
import torch
import numpy as np
from torch import nn
from models import Encoder, Decoder
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from data import Vocabulary, get_loader

LOG_FREQ = 10
SAVE_FREQ = 100

def train_models():
    # Logs
    train_losses = []
    val_losses = []
    val_bleus = []
    best_bleu = float("-INF")

    # Hyperparameters
    cfg = { 
          'device' : "cuda" if torch.cuda.is_available() else "cpu",
          'batch_size' : 32,
          'num_epochs' : 10,
          'lr' : 0.001,
          "momentum": 0.01,
          'hidden_size': 512,
          'embed_size': 256,
          'n_layers': 1,
          'dropout': 0.5,
          'seed' : 0,
          }

    # Data loaders
    train_loader = get_loader("TRAIN", cfg["batch_size"])
    val_loader = get_loader("VAL", cfg["batch_size"])

    # Models
    encoder = Encoder(cfg["embed_size"]).to(cfg["device"])
    decoder = Decoder(cfg["embed_size"], cfg["hidden_size"], len(train_loader.dataset.vocab)).to(cfg["device"])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(cfg["device"])
    params = (list(filter(lambda p: p.requires_grad, encoder.parameters()))+
            list(filter(lambda p: p.requires_grad, decoder.parameters())))
    optimizer = torch.optim.Adam(params, lr=cfg["lr"])

    start = time.time()
    for epoch in range(cfg["num_epochs"]):
        train_loss = train( loader=train_loader,
                            encoder=encoder,
                            decoder=decoder,
                            criterion=criterion,
                            opt=optimizer,
                            epoch=epoch,
                            cfg=cfg)
        train_losses.append(train_loss)
        
        val_loss, val_bleu = validate(  loader=val_loader,
                                        encoder=encoder,
                                        decoder=decoder,
                                        criterion=criterion,
                                        epoch=epoch,
                                        cfg=cfg)
        val_losses.append(val_loss)
        val_bleus.append(val_bleu)
        
        if val_bleu > best_bleu:
            print("Validation BLEU improved from {} to {}".format(best_bleu, val_bleu))
            best_bleu = val_bleu
            print("Saving checkpoint...")
            filename = os.path.join("./data/models", "best-model-{}.ckpt".format(best_bleu))
            torch.save({"encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "best_bleu": best_bleu,
                        "val_bleus": val_bleus,
                        "epoch": epoch
                    }, filename)
        else:
            print("Validation BLEU did not improve")
            print("Saving checkpoint...")
            filename = os.path.join("./data/models", "model-{}.ckpt".format(epoch))
            torch.save({"encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "best_bleu": best_bleu,
                        "val_bleus": val_bleus,
                        "epoch": epoch
                    }, filename)
        print ("Epoch [%d/%d] runtime %ds" % (epoch, cfg["num_epochs"], time.time() - start))

        if epoch > 6:
            if early_stopping(val_bleus):
                break
        start = time.time()
    return

def train(loader, encoder, decoder, criterion, opt, epoch, cfg, start_loss=0.0):

    encoder.train()
    decoder.train()

    start = time.time()
    total_steps = len(loader)
    total_loss = start_loss

    for i, (imgs, caps) in enumerate(loader, start=1):
        
        imgs = imgs.to(cfg["device"])
        caps = caps.to(cfg["device"])
      
        # Forward pass
        features = encoder(imgs)
        outputs = decoder(features, caps)

        # Loss and backward pass
        loss = criterion(outputs.view(-1, len(loader.dataset.vocab)), caps.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        
        if i % LOG_FREQ == 0:
            print('TRAIN: Epoch [{}], Step [{}/{}], Time {}, Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, i, total_steps, time.time() - start, loss.item(), np.exp(loss.item())))

        if i % SAVE_FREQ == 0:
            print("Saving checkpoint...")
            filename = os.path.join("./data/models/checkpoints/train-model-{}-{}.ckpt".format(epoch, i))
            torch.save({"encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "optimizer" : opt.state_dict(),
                        "total_loss": total_loss,
                        "epoch": epoch,
                        "train_step": i,
                    }, filename)

    return total_loss/total_steps
            
def validate(loader, encoder, decoder, criterion, epoch, cfg, start_loss=0.0, start_bleu=0.0):

    encoder.eval()
    decoder.eval()
    
    start = time.time()
    total_steps = len(loader)
    total_loss = start_loss
    total_bleu = start_bleu

    with torch.no_grad():
        for i, (imgs, caps, all_caps) in enumerate(loader, start=1):
            
            imgs = imgs.to(cfg["device"])
            caps = caps.to(cfg["device"])

            # Forward pass
            features = encoder(imgs)
            outputs = decoder(features, caps)

            # Loss
            loss = criterion(outputs.view(-1, len(loader.dataset.vocab)), caps.view(-1))
            total_loss += loss.item()

            # BLEU
            _, predicted = outputs.max(2)
            candidates = predicted.to("cpu").numpy()
            all_caps = all_caps.to("cpu").numpy()

            batch_bleu_score = bleu_eval(candidates, all_caps, loader.dataset.vocab)
            total_bleu += batch_bleu_score

            if i % LOG_FREQ == 0:
                print('VAL: Epoch [{}], Step [{}/{}], Time {}, Loss: {:.4f}, Perplexity: {:5.4f}, BLEU: {:5.4f}'.format(epoch, i, total_steps, time.time() - start, loss.item(), np.exp(loss.item()), batch_bleu_score))

            if i % SAVE_FREQ == 0:
                print("Saving checkpoint...")
                filename = os.path.join("./data/models/checkpoints/val-model-{}-{}.ckpt".format(epoch, i))
                torch.save({"encoder": encoder.state_dict(),
                            "decoder": decoder.state_dict(),
                            "total_loss": total_loss,
                            "total_bleu": total_bleu,
                            "epoch": epoch,
                            "val_step": i,
                        }, filename)

    return total_loss/total_steps, total_bleu/total_steps

def bleu_eval(candidates, all_caps, v):
    candidates = candidates.tolist()
    references = []
    for batch in all_caps:
        batch_ref = []
        for cap in batch:
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
    # If the best BLEU is in the patience range and it hasn't converged yet
    if best_bleu in last_bleus and not len(set(last_bleus)) == 1:
        return False
    # If the best BLEU hasn't converged yet (ie it's a peak)
    elif best_bleu not in val_bleus[:len(val_bleus)-patience]:
        return False
    else:
        return True

    return True