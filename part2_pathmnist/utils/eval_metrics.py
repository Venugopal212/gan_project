import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision.transforms import Resize, Normalize, Compose
import torchvision.transforms.functional as TF

def get_evaluation_metrics(device="cuda"):
    fid = FrechetInceptionDistance(feature=64).to(device)
    is_score = InceptionScore().to(device)
    kid = KernelInceptionDistance(subset_size=50).to(device)
    return fid, is_score, kid

def preprocess_for_eval(imgs, resolution=299):
    """Resize and normalize images for Inception model input (IS, FID, KID)"""
    resized = TF.resize(imgs, [resolution, resolution])
    return (resized + 1) / 2  # Rescale from [-1,1] to [0,1]
