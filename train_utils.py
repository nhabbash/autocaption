import time
import json
import os
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import sampler
from models import Encoder, Decoder
from data import Vocabulary, get_loader
from utils import get_caption, bleu_eval, early_stopping
from fastprogress.fastprogress import progress_bar, master_bar

LOG_FREQ = 10
SAVE_FREQ = 100

experiment_folder = "./data/experiments/"+time.strftime("%Y-%m-%d")

standard_cfg = { 
        "batch_size" : 32,
        "num_epochs" : 15,
        "lr" : 0.001,
        "momentum": 0.01,
        "hidden_size": 512,
        "embed_size": 512,
        "n_layers": 1,
        "dropout": 0.1,
        "seed" : 0,
        "dataset": "flickr8k"
        }

def train_models(cfg=standard_cfg, checkpoint=None):

    if checkpoint:
        chk_epoch = checkpoint["epoch"]
        date = checkpoint["date"]
        set_experiment_folder(date)
        filename = os.path.join(experiment_folder, "/models/model-{}.ckpt".format(chk_epoch))
        epoch_checkpoint = torch.load(filename)

        # Loading weights and histories
        encoder.load_state_dict(epoch_checkpoint["encoder"])
        decoder.load_state_dict(epoch_checkpoint["decoder"])
        optimizer.load_state_dict(epoch_checkpoint["optimizer"])
        epoch = epoch_checkpoint["epoch"]

        train_losses = epoch_checkpoint["train_losses"]
        val_losses = epoch_checkpoint["val_losses"]
        val_bleus = epoch_checkpoint["val_bleus"]
        best_bleu = epoch_checkpoint["val_bleu"]
    else:
        # Setting experiments folder
        set_experiment_folder(time.strftime("%Y-%m-%d"))
        # Saving hyperparameters to json
        with open(experiment_folder+"/hyperparameters.json", "w", encoding='utf-8') as f:
            json.dump(cfg, f)

        # Logs
        train_losses = []
        val_losses = []
        train_perplexities = []
        val_perplexities = []
        val_bleus = []
        best_bleu = float("-INF")
        best_epoch = 0

        # Data loaders
        train_loader = get_loader("TRAIN", cfg["batch_size"])
        val_loader = get_loader("VAL", cfg["batch_size"])

        # Models
        encoder = Encoder(cfg["embed_size"], cfg["momentum"])
        decoder = Decoder(cfg["embed_size"], 
                        cfg["hidden_size"], 
                        len(train_loader.dataset.vocab),
                        cfg["n_layers"],
                        cfg["dropout"])

        params = (list(filter(lambda p: p.requires_grad, encoder.parameters()))+
            list(filter(lambda p: p.requires_grad, decoder.parameters())))
        optimizer = torch.optim.Adam(params, lr=cfg["lr"])

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Cast to device
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.to(cfg["device"])
    decoder.to(cfg["device"])
    criterion.to(cfg["device"])

    # Progress bar
    mb = master_bar(range(cfg["num_epochs"]))

    # Epoch loop
    for epoch in mb:
        start = time.time()

        train_loss = train( loader=train_loader,
                            encoder=encoder,
                            decoder=decoder,
                            criterion=criterion,
                            opt=optimizer,
                            epoch=epoch,
                            cfg=cfg,
                            mb=mb)
        train_losses.append(train_loss)
        train_perplexity = np.exp(train_loss)
        train_perplexities.append(train_perplexity)
        
        val_loss, val_bleu = validate(  loader=val_loader,
                                        encoder=encoder,
                                        decoder=decoder,
                                        criterion=criterion,
                                        epoch=epoch,
                                        cfg=cfg,
                                        mb=mb)
        val_losses.append(val_loss)
        val_bleus.append(val_bleu)
        val_perplexity = np.exp(val_loss)
        val_perplexities.append(val_perplexity)

        mb.write('> Epoch {}/{}'.format(epoch + 1, cfg["num_epochs"]))
        mb.write('# TRAIN')
        mb.write('# Loss {:.3f}, Perplexity {:.3f}'.format(train_loss, train_perplexity))
        mb.write('# VALIDATION')
        mb.write('# Loss {:.3f}, Perplexity {:.3f}, BLEU {:.3f}'.format(val_loss, val_perplexity, val_bleu))
        mb.write(">Runtime {:.3f}".format(time.time() - start))
        
        if val_bleu > best_bleu:
            mb.write("## Validation BLEU improved from {} to {} in EPOCH {}".format(best_bleu, val_bleu, epoch))
            best_bleu = val_bleu
            best_epoch = epoch

        mb.write("## Saving model at {}/models/model-{}.ckpt".format(experiment_folder, epoch))
        filename = os.path.join(experiment_folder+"/models/", "model-{}.ckpt".format(epoch))
        torch.save({"encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_perplexities": train_perplexities,
                    "val_perplexities": val_perplexities,
                    "best_bleu": best_bleu,
                    "val_bleus": val_bleus,
                    "epoch": epoch,
                    "best": best_epoch
                }, filename)
            
        if epoch > 5:
            if early_stopping(val_bleus, patience=3):
                mb.write("## Validation BLEU did not improve for 3 consecutive epochs, stopping")
                break
    

    return {
        "train_losses": train_losses, 
        "val_losses": val_losses,
        "val_bleus": val_bleus, 
        "best_bleu": best_bleu, 
        "train_perplexities": train_perplexities,
        "val_perplexities": val_perplexities}

def train(loader, encoder, decoder, criterion, opt, epoch, cfg, start_step=1, start_loss=0.0, save_checkpoints=False, mb=None):
    encoder.train()
    decoder.train()

    total_steps = math.ceil(len(loader.dataset)/loader.batch_sampler.batch_size)
    total_loss = start_loss

    if mb:
        bar = progress_bar(range(start_step, total_steps+1), parent=mb)
    else:
        bar = range(start_step, total_steps+1)
    
    for step in bar:
        # Sample a different caption length for each step
        indices = loader.dataset.get_indices()
        length_sampler = sampler.SubsetRandomSampler(indices=indices)
        loader.batch_sampler.sampler = length_sampler
        
        loader_iter = iter(loader)
        imgs, caps = next(loader_iter)
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
        
        if mb:
            mb.child.comment = 'Epoch {}, Loss {:.3f}, Perplexity {:.3f}'.format(epoch + 1, loss.item(), np.exp(loss.item()))

    return total_loss/total_steps

def validate(loader, encoder, decoder, criterion, epoch, cfg, start_step=1, start_loss=0.0, start_bleu=0.0, mb=None):
    encoder.eval()
    decoder.eval()
    
    start = time.time()
    total_steps = math.ceil(len(loader.dataset)/loader.batch_sampler.batch_size)
    total_loss = start_loss
    total_bleu = start_bleu

    if mb:
        bar = progress_bar(range(start_step, total_steps+1), parent=mb)
    else:
        bar = range(start_step, total_steps+1)

    with torch.no_grad():
        for step in bar:
            # Sample a different caption length for each step
            indices = loader.dataset.get_indices()
            length_sampler = sampler.SubsetRandomSampler(indices=indices)
            loader.batch_sampler.sampler = length_sampler

            loader_iter = iter(loader)
            imgs, caps, all_caps = next(loader_iter)
            
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

            batch_bleu_score = bleu_eval(candidates.tolist(), all_caps, loader.dataset.vocab)
            total_bleu += batch_bleu_score

            if mb:
                mb.child.comment = 'Epoch {}, Loss {:.3f}, Perplexity {:.3f}, BLEU {:.3f}'.format(epoch + 1, loss.item(), np.exp(loss.item()), batch_bleu_score)

    return total_loss/total_steps, total_bleu/total_steps

def test(loader, encoder, decoder, cfg, sample="beam"):
    encoder.eval()
    decoder.eval()
    
    results = []

    progress = progress_bar(loader)

    with torch.no_grad():
        for (orig_image, image, caption, all_caps) in progress:

            image = image.to(cfg["device"])
            caption = caption.to(cfg["device"])

            candidates = get_caption(image, 
                                    encoder, 
                                    decoder, 
                                    loader.dataset.vocab,
                                    sample,
                                    False)

            # BLEU
            all_caps = all_caps.to("cpu").numpy()
            
            # For Beam search evaluate only the first
            if sample == "beam":
                candidates = [candidates[0]]

            bleu_score = bleu_eval(candidates, all_caps, loader.dataset.vocab)
            decoded_cap = [decode_cap(c, loader.dataset.vocab) for c in candidates]

            decoded_all_caps = [[decode_cap(c.flatten().tolist(), loader.dataset.vocab)] for c in all_caps]

            results.append({"orig_image": orig_image, 
                            "caption": decoded_cap,
                            "all_caps": decoded_all_caps,
                            "bleu": bleu_score})

    results = sorted(results, key=lambda k: k['bleu']) 
    return results

def set_experiment_folder(date):
    experiment_folder = "./data/experiments/"+date
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder + "/models")
