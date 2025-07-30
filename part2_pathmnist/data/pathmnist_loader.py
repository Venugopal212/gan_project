import torch
from medmnist import PathMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

def get_pathmnist_loaders(batch_size=128, root="./data/medmnist"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train_ds = PathMNIST(split='train', root=root, transform=transform, download=True)
    test_ds = PathMNIST(split='test', root=root, transform=transform, download=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
