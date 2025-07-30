import torch
from torch import nn

class CGANGenerator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_shape):
        super(CGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.img_shape = img_shape
        self.input_dim = latent_dim + n_classes

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.input_dim, 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Ensure labels are one-hot and 2D
        if labels.dim() > 2:
            labels = labels.view(labels.size(0), -1)
        elif labels.dim() == 1:
            labels = nn.functional.one_hot(labels, num_classes=self.n_classes).float()
        elif labels.size(1) != self.n_classes:
            labels = nn.functional.one_hot(labels.squeeze(), num_classes=self.n_classes).float()

        # Concatenate noise and label
        x = torch.cat([z, labels], dim=1)  # (B, latent_dim + n_classes)

        out = self.model(x)
        img = out.view(out.size(0), *self.img_shape)
        return img
