import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    """
    Pretrained CNN
    """
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        # Load pretrained ResNet
        resnet = models.resnet50(pretrained=True)

        # Delete the last FC and pooling layers
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.fine_tune()

    def forward(self, images):
        # Forward pass, out=(batch_size, embed_dim)
        out = self.resnet(images)
        out = self.embed(out.reshape(out.size(0), -1))
        out = self.bn(out)
        return out

    def fine_tune(self, fine_tune=False):
        for p in self.resnet.parameters():
            p.requires_grad = False

        # Only fine-tune the last 2 convolutions
        for c in list(self.resnet.children())[7:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


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
        input = features.unsqueeze(1)
        for i in range(max_len):
            hiddens, states = self.lstm(input, states)
            out = self.linear(hiddens.squeeze(1))
            # Most likely encoded word
            predicted = out.argmax(1)
            sample.append(predicted.item())
            input = self.embed(predicted)
            input = input.unsqueeze(1)
        return sample

    def sample_beam_search(self, inputs, states=None, max_len=20, beam_width=5):
        return