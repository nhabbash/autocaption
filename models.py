import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    """
    Pretrained CNN
    """

    def __init__(self, encoded_size=14):
        super(Encoder, self).__init__()
        
        # Load pretrained ResNet
        resnet = models.resnet50(pretrained=True)

        # Delete the last FC and pooling layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Adaptive pooling to allow input images of variable sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_size, encoded_size))

        self.fine_tune()

    def forward(self, images):
        # Forward pass, out=(batch_size, 2048, encoded_size, encoded_size)
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        return out

    def fine_tune(self, fine_tune=False):
        for p in self.resnet.parameters():
            p.requires_grad = False

        # Only fine-tune layers 5
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


