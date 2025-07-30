import torch
from torch import nn

class CGANDiscriminator(nn.Module):
    def __init__(self, n_classes, img_shape):
        super(CGANDiscriminator, self).__init__()
        self.n_classes = n_classes
        self.img_shape = img_shape
        self.input_dim = int(torch.prod(torch.tensor(img_shape))) + n_classes

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Flatten the image to (B, C*H*W)
        img_flat = img.view(img.size(0), -1)

        # Ensure labels are one-hot and 2D
        if labels.dim() > 2:
            labels = labels.view(labels.size(0), -1)
        elif labels.dim() == 1:
            labels = nn.functional.one_hot(labels, num_classes=self.n_classes).float()
        elif labels.size(1) != self.n_classes:
            labels = nn.functional.one_hot(labels.squeeze(), num_classes=self.n_classes).float()

        # Concatenate image and label
        x = torch.cat([img_flat, labels], dim=1)  # (B, C*H*W + n_classes)

        validity = self.model(x)
        return validity
