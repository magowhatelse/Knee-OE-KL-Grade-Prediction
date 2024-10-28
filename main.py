import pandas as pd
import os
from torch.utils.data import DataLoader

from args import get_args
from dataset import knee_X_Ray_dataset


# train val loader, call trainer

def main():

    # 1. step: We need some arguments
    args = get_args()

    # 2. step: iterate over folds
    for fold in range(5):
        print("Fold: ",fold)

        train_set = pd.read_csv(os.path.join(args.csv_dir ,f"fold_{fold}_train.csv")) # fold_0_train.csv
        val_set = pd.read_csv(os.path.join(args.csv_dir ,f"fold_{fold}_val.csv"))  # f"" ??

    # 3. step: load dataset
    train_dataset = knee_X_Ray_dataset(dataset=train_set)
    val_dataset = knee_X_Ray_dataset(dataset=val_set)


    # 4. step: create data loaders
    train_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,  shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,batch_size=args.batch_size,  shuffle=False)

    # 5. step: init the model
    train_model = ""

    # 6. step: train the model 


    # 7. step: evaluate the model



if "__main__":
    main()