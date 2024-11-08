import numpy as np
import cv2
from torch.utils.data import Dataset
import torch

def read_xray(path_file):
    xray = cv2.imread(path_file)  #, cv2.IMREAD_GRAYSCALE)

    xray = xray.astype(np.float32) / 255.0
    ret = np.empty((3, xray.shape[0], xray.shape[1]), dtype=xray.dtype)
    ret[0] = xray[:, :, 0]
    ret[1] = xray[:, :, 1]
    ret[2] = xray[:, :, 2]

    return ret


class knee_Xray_dataset(Dataset):
    def __init__(self, dataset, normalize=True, mean=[0.6190, 0.6190, 0.6190], std=[0.1530, 0.1530, 0.1530]):
        self.dataset = dataset
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.dataset)

    def normalize_image(self, img):
        # normalize each channel
        for i in range(3):
            img[i] = (img[i] - self.mean[i]) / self.std[i]
        return img

    def __getitem__(self, idx):
        img = read_xray(self.dataset['Path'].iloc[idx])
        
        if self.normalize:
            img = self.normalize_image(img)
            
        target = self.dataset['KL'].iloc[idx]
        name = self.dataset['Name'].iloc[idx]
        
        res = {
            'Name': name,
            'img': img,
            'target': target
        }

        return res