import time
import os
import torch
import math
import numpy as np
from torch import nn
from torch.utils.data import Dataset, sampler
from models import Encoder, Decoder
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from data import Vocabulary, get_loader
from utils import get_caption
from fastprogress.fastprogress import progress_bar, master_bar
import wandb

LOG_FREQ = 10
SAVE_FREQ = 100

def train_sweep():
    
    cfg_defaults = { 
        "device" : "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size" : 32,
        "num_epochs" : 50,
        "lr" : 0.001,
        "momentum": 0.01,
        "hidden_size": 512,
        "embed_size": 512,
        "n_layers": 1,
        "dropout": 0.1,
        "seed" : 0,
        "dataset": "flickr8k"
        }
    
    # Wandb project init
    wandb.init(
    project="autocaption",
    notes="sweep",
    tags=["sweep"],
    config=cfg_defaults,
    )
    
    # Config variable for sweeps
    cfg = wandb.config

    # Logs
    train_losses = []
    val_losses = []
    val_bleus = []
    best_bleu = float("-INF")

    # Data loaders
    train_loader = get_loader("TRAIN", cfg["batch_size"])
    val_loader = get_loader("VAL", cfg["batch_size"])

    # Models
    encoder = Encoder(cfg["embed_size"], cfg["momentum"]).to(cfg["device"])
    decoder = Decoder(cfg["embed_size"], 
                    cfg["hidden_size"], 
                    len(train_loader.dataset.vocab),
                    cfg["n_layers"],
                    cfg["dropout"]).to(cfg["device"])
    wandb.watch([encoder, decoder])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(cfg["device"])
    params = (list(filter(lambda p: p.requires_grad, encoder.parameters()))+
            list(filter(lambda p: p.requires_grad, decoder.parameters())))
    optimizer = torch.optim.Adam(params, lr=cfg["lr"])

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
        
        

        val_loss, val_bleu = validate(  loader=val_loader,
                                        encoder=encoder,
                                        decoder=decoder,
                                        criterion=criterion,
                                        epoch=epoch,
                                        cfg=cfg,
                                        mb=mb)
        val_losses.append(val_loss)
        val_bleus.append(val_bleu)
        #print('> Epoch {}/{}'.format(epoch + 1, cfg["num_epochs"]))
        mb.write('> Epoch {}/{}'.format(epoch + 1, cfg["num_epochs"]))
        mb.write('TRAIN')
        mb.write('# Loss {:.3f}, Perplexity {:.3f}'.format(train_loss, np.exp(train_loss)))
        mb.write('VALIDATION')
        mb.write('# Loss {:.3f}, Perplexity {:.3f}, BLEU {:.3f}'.format(val_loss, np.exp(val_loss), val_bleu))
        mb.write(">Runtime {:.3f}".format(time.time() - start))

        # Send logs to wandb
        wandb.log({'train_loss': train_loss,
                'train_perplexity': np.exp(train_loss),
                'val_loss': val_loss,
                'val_perplexity': np.exp(val_loss),
                'val_bleu': val_bleu}, step=epoch)
        
        if val_bleu > best_bleu:
            mb.write("Validation BLEU improved from {} to {}, saving model at ./data/models/sweep1/best-model.ckpt".format(best_bleu, val_bleu))
            best_bleu = val_bleu
            
            filename = os.path.join("./data/models", "best-model.ckpt")
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
            mb.write("Validation BLEU did not improve, saving model at ./data/models/sweep1/model-{}.ckpt".format(epoch))
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
        
        # Saving last model to wandb, works only if Jupyter is executed as Admin
        try: 
            wandb.save(filename)
        except:
            pass
            
        if epoch > 5:
            if early_stopping(val_bleus, patience=3):
                mb.write("Validation BLEU did not improve for 3 consecutive epochs, stopping")
                break

def train_models(cfg):

    # Logs
    train_losses = []
    val_losses = []
    val_bleus = []
    best_bleu = float("-INF")

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
        
        

        val_loss, val_bleu = validate(  loader=val_loader,
                                        encoder=encoder,
                                        decoder=decoder,
                                        criterion=criterion,
                                        epoch=epoch,
                                        cfg=cfg,
                                        mb=mb)
        val_losses.append(val_loss)
        val_bleus.append(val_bleu)
        
        mb.write('> Epoch {}/{}'.format(epoch + 1, cfg["num_epochs"]))
        mb.write('TRAIN')
        mb.write('# Loss {:.3f}, Perplexity {:.3f}'.format(train_loss, np.exp(train_loss)))
        mb.write('VALIDATION')
        mb.write('# Loss {:.3f}, Perplexity {:.3f}, BLEU {:.3f}'.format(val_loss, np.exp(val_loss), val_bleu))
        mb.write(">Runtime {:.3f}".format(time.time() - start))
        
        if val_bleu > best_bleu:
            mb.write("Validation BLEU improved from {} to {}, saving model at ./data/models/best-model.ckpt".format(best_bleu, val_bleu))
            best_bleu = val_bleu
            
            filename = os.path.join("./data/models", "best-model.ckpt")
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
            mb.write("Validation BLEU did not improve, saving model at ./data/models/model-{}.ckpt".format(epoch))
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
            
        if epoch > 5:
            if early_stopping(val_bleus, patience=3):
                mb.write("Validation BLEU did not improve for 3 consecutive epochs, stopping")
                break
    return train_losses, val_losses, val_bleus

def train(loader, encoder, decoder, criterion, opt, epoch, cfg, start_step=1, start_loss=0.0, save_checkpoints=False, mb=None):
    encoder.train()
    decoder.train()

    total_steps = math.ceil(len(loader.dataset)/loader.batch_sampler.batch_size)
    total_loss = start_loss
    bar = progress_bar(range(start_step, total_steps+1), parent=mb)
    
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
        
        mb.child.comment = 'Epoch {}, Loss {:.3f}, Perplexity {:.3f}'.format(epoch + 1, loss.item(), np.exp(loss.item()))

    return total_loss/total_steps

def validate(loader, encoder, decoder, criterion, epoch, cfg, start_step=1, start_loss=0.0, start_bleu=0.0, save_checkpoints=False, mb=None):
    encoder.eval()
    decoder.eval()
    
    start = time.time()
    total_steps = math.ceil(len(loader.dataset)/loader.batch_sampler.batch_size)
    total_loss = start_loss
    total_bleu = start_bleu

    bar = progress_bar(range(start_step, total_steps+1), parent=mb)

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

            mb.child.comment = 'Epoch {}, Loss {:.3f}, Perplexity {:.3f}, BLEU {:.3f}'.format(epoch + 1, loss.item(), np.exp(loss.item()), batch_bleu_score)

    return total_loss/total_steps, total_bleu/total_steps

# For one training cycle
def resume_training(checkpoint_epoch, checkpoint_step, cfg):
    filename = os.path.join("./data/models", "train-model-{}-{}.ckpt".format(checkpoint_epoch, checkpoint_step))
    train_checkpoint = torch.load(filename)

    filename = os.path.join("./data/models", "model-{}.ckpt".format(checkpoint_epoch))
    epoch_checkpoint = torch.load(filename)

    filename = os.path.join("./data/models", "best-model.ckpt")
    best_checkpoint = torch.load(filename)

    # Loading weights and histories
    encoder.load_state_dict(train_checkpoint["encoder"])
    decoder.load_state_dict(train_checkpoint["decoder"])
    optimizer.load_state_dict(train_checkpoint["optimizer"])
    epoch = train_checkpoint["epoch"]

    train_losses = epoch_checkpoint["train_losses"]
    val_losses = epoch_checkpoint["val_losses"]
    val_bleus = epoch_checkpoint["val_bleus"]
    best_bleu = best_checkpoint["val_bleu"]

    train_loss = train( loader=train_loader,
                        encoder=encoder,
                        decoder=decoder,
                        criterion=criterion,
                        opt=optimizer,
                        epoch=epoch,
                        cfg=cfg)
    train_losses.append(train_loss)

    return train_losses

# For one validation cycle
def resume_validation(checkpoint_epoch, checkpoint_step, cfg):
    filename = os.path.join("./data/models", "val-model-{}-{}.ckpt".format(checkpoint_epoch, checkpoint_step))
    train_checkpoint = torch.load(filename)

    filename = os.path.join("./data/models", "model-{}.ckpt".format(checkpoint_epoch))
    epoch_checkpoint = torch.load(filename)

    filename = os.path.join("./data/models", "best-model.ckpt")
    best_checkpoint = torch.load(filename)

    # Loading weights and histories
    encoder.load_state_dict(train_checkpoint["encoder"])
    decoder.load_state_dict(train_checkpoint["decoder"])
    optimizer.load_state_dict(train_checkpoint["optimizer"])
    epoch = train_checkpoint["epoch"]

    train_losses = epoch_checkpoint["train_losses"]
    val_losses = epoch_checkpoint["val_losses"]
    val_bleus = epoch_checkpoint["val_bleus"]
    best_bleu = best_checkpoint["val_bleu"]

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
        filename = os.path.join("./data/models", "best-model.ckpt")
        torch.save({"encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "best_bleu": best_bleu,
                    "val_bleus": val_bleus,
                    "epoch": epoch
                }, filename)

    return val_losses, val_bleus

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
    if np.std(last_bleus) < 0.05:
        return True

    if best_bleu in last_bleus:
        if best_bleu not in val_bleus[:len(val_bleus)-patience]:
            return False
        else:
            return True

    return True

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

