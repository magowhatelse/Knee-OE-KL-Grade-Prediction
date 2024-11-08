import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms

from dataset import knee_Xray_dataset
from model import MyModel
from trainer import train_model, validation_model
from args import get_args


def main():
    # 1. We need some arguments
    args = get_args()

    # 2. Iterate through the folds
    for fold in range(5):
        print('Fold: ', fold)
        a = args.csv_dir

        train_set = pd.read_csv(os.path.join(args.csv_dir, fr'fold_{fold}_train.csv'))  # fold_0_train.csv ... fold_3_train.csv fold_4_train.csv
        val_set = pd.read_csv(os.path.join(args.csv_dir,fr'fold_{fold}_val.csv'))  # fold_0_val.csv ... fold_3_val.csv fold_4_val.csv

        # 3. Preparing datasets
        train_dataset = knee_Xray_dataset(dataset=train_set) 
        val_dataset = knee_Xray_dataset(dataset=val_set)

        # 4. Creat data loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=8, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=8, pin_memory=torch.cuda.is_available())

        # 5. Initialize the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MyModel(backbone="resnet34").to(device)

        # 6. Train the model
        train_model(model, train_loader, val_loader, args, fold)
        
    

        # 7. Evaluate the model
        # validation_model(model, val_loader, args)



if __name__ == '__main__':
    main()


