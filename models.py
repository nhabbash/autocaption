import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    """
    Pretrained CNN
    """

    def __init__(self, embed_size, encoded_size=14):
        super(Encoder, self).__init__()
        
        # Load pretrained ResNet
        resnet = models.resnet50(pretrained=True)

        # Delete the last FC and pooling layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Adaptive pooling to allow input images of variable sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_size, encoded_size))

        self.linear = nn.Linear(2048*encoded_size*encoded_size, embed_size)

        self.fine_tune()

    def forward(self, images):
        # Forward pass, out=(batch_size, features_dim)
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        out = self.linear(out.reshape(out.size(0), -1))
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

    def __init__(self, max_len, vocab_size, embedding_dim, num_layers, hidden_size, encoder_dim=2048, dropout=0.5):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_len = max_len

    def forward(self, features, captions, lengths):
        # Forward pass, out=(cap_len, vocab_size)
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1) #Check features size
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        unpacked = pad_packed_sequence(hiddens, batch_first=True)
        out = self.linear(unpacked[0])
        return out

    def sample(self, features, states=None):
        # Caption generation using greedy search
        sample = []

        input = features.unsqueeze(1)
        for i in range(self.max_len):
            out, states = self.lstm(input, states)
            out = self.linear(out.squeeze(1))
            _, predicted = out.max(1)
            sample.append(predicted)
            input = self.embed(predicted)
            input = input.unsqueeze(1)

        sample = torch.stack(sample, 1)
        return sample