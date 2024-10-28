import cv2 as cv
from torch.utils.data import Dataset

def read_xray(path):
    xray = cv.imread(path)


    ## maybe ROI in one?
    return xray


class knee_X_Ray_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img = read_xray(self.dataset["Path"].iloc[idx])
        target = read_xray(self.dataset["KL"].iloc[idx])

        res = {
            "Name" : self.dataset["Name"].iloc[idx],
            "img" : img,
            "target" : target
        }

        return res
