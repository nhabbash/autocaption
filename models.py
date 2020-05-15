import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    """
    Pretrained CNN
    """
    def __init__(self, embed_size, momentum, tune=False):
        super(Encoder, self).__init__()

        self.tune = tune
        # Load pretrained ResNet
        resnet = models.resnet101(pretrained=True)

        # Delete the last FC and pooling layers
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=momentum)
        self.fine_tune()

    def forward(self, images):
        # Forward pass, out=(batch_size, embed_dim)
        out = self.resnet(images)
        out = self.embed(out.reshape(out.size(0), -1))
        out = self.bn(out)
        return out

    def fine_tune(self):
        for p in self.resnet.parameters():
            p.requires_grad = False

        # Only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = self.tune


class Decoder(nn.Module):
    """
    LSTM RNN
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # Forward pass, out=(batch_size, captions.shape[1], vocab_size)
        captions = captions[:,:-1] 
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        out = self.linear(hiddens)
        return out

    def sample(self, inputs, states=None, max_len=20):
        # Caption generation using greedy search
        sample = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            out = self.linear(hiddens.squeeze(1))
            # Most likely encoded word
            predicted = out.argmax(1)
            sample.append(predicted.item())
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sample

    def sample_beam_search(self, inputs, states=None, max_len=20, beam_width=5):
        # Caption generation using beam search

        idx_sequences = [[[], 0.0, inputs, states]]

        for _ in range(max_len):
            all_candidates = []
            for idx_seq in idx_sequences:
                hiddens, states = self.lstm(idx_seq[2], idx_seq[3])
                outputs = self.linear(hiddens.squeeze(1))
                # Transform to log prob to have larger probabilities
                log_probs = nn.functional.log_softmax(outputs, -1)
                top_log_probs, top_idx = log_probs.topk(beam_width, 1)
                top_idx = top_idx.squeeze(0)

                for i in range(beam_width):
                    next_idx_seq, log_prob = idx_seq[0][:], idx_seq[1]
                    next_idx_seq.append(top_idx[i].item())
                    log_prob += top_log_probs[0][i].item()
                    
                    inputs = self.embed(top_idx[i].unsqueeze(0)).unsqueeze(0)
                    all_candidates.append([next_idx_seq, log_prob, inputs, states])

            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            idx_sequences = ordered[:beam_width]

        return [idx_seq[0] for idx_seq in idx_sequences]