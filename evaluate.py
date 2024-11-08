import os 
import test
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from zmq import device
from dataset import knee_Xray_dataset
from model import MyModel
from trainer import train_model, validation_model
from args import get_args
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, average_precision_score, ConfusionMatrixDisplay

import torch.optim as optim

args = get_args()

test_set = pd.read_csv(r"/home/user/persistent/KL Grade Prediction/data/CSV/test_data.csv")

test_dataset = knee_Xray_dataset(dataset=test_set)

test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=8, pin_memory=torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

folds_y_pred = []
folds_y_true = []
for fold in range(5):
    print('Fold: ', fold)
    a = args.csv_dir
    
    path_weights_fold = os.path.join(args.out_dir, fr'fold_{fold}.pth')

    model = MyModel().to(device)
    model.load_state_dict(torch.load(path_weights_fold))


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.eval()

    y_pred = []
    y_true = []

    test_running_loss = 0.0
    correct = 0
    total = 0 
    for batch in test_loader:
        inputs = batch["img"].to(device)
        targets = batch["target"].to(device)
        outputs = model(inputs)
        a = outputs.detach().cpu().numpy()
            
        outputs_softmax = torch.softmax(outputs, dim=1)
        _,predicted = torch.max(outputs_softmax, 1)

        total += targets.size(0)
        correct += (preds == targets).sum().item()

        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(targets.cpu().numpy())

        loss = criterion(outputs,targets)
        test_running_loss += loss.item()

    print(f'Accuracy of the network: {100 * correct // total} %')

    folds_y_pred.extend(y_pred)
    folds_y_true.extend(y_true)

    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred_soft, multi_class="ovo", average="macro")
    precison = average_precision_score(y_true, y_pred_soft, average="macro")

    print(f"Balanced Accuracy: {balanced_accuracy}, ROC-AUC-Score: {roc},  Precison: {precision}")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()




