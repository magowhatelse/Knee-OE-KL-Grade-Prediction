from tqdm import tqdm
from dataset import knee_Xray_dataset
import pandas as pd
import os
from torch.utils.data import DataLoader
from args import get_args
import torch 
args = get_args()

train_set = pd.read_csv(os.path.join(args.csv_dir, "data.csv"))
train_dataset = knee_Xray_dataset(dataset=train_set)

from torch.utils.data import DataLoader
import numpy as np

loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

mean = 0.
std = 0.
n_samples = 0.

for images in loader:
    images = images["img"]
    images = images.view(images.size(0), images.size(1), -1)  # Flatten H x W into a vector
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    n_samples += images.size(0)

mean /= n_samples
std /= n_samples

print(f"Mean: {mean}")
print(f"Std: {std}")

#Mean: tensor([0.6190, 0.6190, 0.6190])
# Std: tensor([0.1530, 0.1530, 0.1530])
