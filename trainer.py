import os
import torch 
import torch.nn as nn
from torch import optim
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, average_precision_score, ConfusionMatrixDisplay
import numpy as np
from torcheval.metrics import MulticlassAUROC, MulticlassAUPRC, MulticlassConfusionMatrix, MulticlassAccuracy
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, val_loader, args, fold):
    # 1. set model to train mode
    model.train()
    model.to(device)
    # 2. define loss function
    criterion = nn.CrossEntropyLoss()
    optimizer  = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    #others: Ada , AdaGrad
    
    # 3. prepare the lists for the results
    train_loss_history = []
    train_balanced_acc_history = []
    train_roc_auc_history = []
    train_avg_precision_history = []


    val_loss_history = []
    val_balanced_acc_history = []
    val_roc_auc_history = []
    val_avg_precision_history = []


    best_balanced_acc = 0.0
    best_model_path = ''

    print(device)
    
    # 4. train loop
    for epoch in range(args.epochs):
        running_loss = 0.0
        bal_accuracy = []
        y_pred = []
        y_true = []
        roc = [] 
        prc = [] 

        for batch in train_loader:
            # get inputst and targets
            inputs = batch["img"].to(device)
            targets = batch["target"].to(device)

            # zero the param gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            labels = torch.argmax(outputs, dim=1)
            outputs_softmax = torch.softmax(outputs, dim=1)
            _,predicted = torch.max(outputs_softmax, 1)

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            

            # calc balanced accuracy
            # balanced_accuracy = balanced_accuracy_score(targets.cpu().numpy(), labels.cpu().numpy())     
            # bal_accuracy += balanced_accuracy
            multicalss_accuracy = MulticlassAccuracy(num_classes=5)
            multicalss_accuracy.update(outputs_softmax, targets)
            bal_accuracy.append(multicalss_accuracy.compute().mean().item())

      
            # calc roc-auc
            train_roc = MulticlassAUROC(num_classes=5, average=None)
            train_roc.update(outputs_softmax, targets)
            roc.append(train_roc.compute().mean().item()) 
            
            # calc avg precision    
            train_avg_precision = MulticlassAUPRC(num_classes=5, average="macro")
            train_avg_precision.update(outputs_softmax, targets)
            prc.append(train_avg_precision .compute().item())
            
            

            # val_confusion_matrix = MulticlassConfusionMatrix(num_classes=5)
            # val_confusion_matrix.update(outputs_softmax, targets)
            # print(f"Confusion Matrix: {val_confusion_matrix.compute()}")
                
            # calc loss
            loss = criterion(outputs,targets)
            running_loss += loss.item()    
            
            
    
        train_loss_history.append(running_loss / len(train_loader))    
        train_avg_precision_history.append(sum(prc) / len(prc))
        train_roc_auc_history.append(sum(roc) / len(roc))
        train_balanced_acc_history.append(balanced_accuracy_score(y_true, y_pred))   


        print("__" * 20, "Epoch:", epoch+1 ,"__" * 20)
        print(f"Loss: {running_loss / len(train_loader)}")
        print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred)}")
        print(f"ROC-AUC: {sum(roc) / len(roc)}")
       
        
        val_loss, val_balanced_accuracy, val_roc, val_precision = validation_model(model, val_loader, args)

        if val_balanced_accuracy > best_balanced_acc:
            best_balanced_acc = val_balanced_accuracy
            best_model_path = f'{args.out_dir}/best_model_fold_{fold}.pth'
            torch.save(model.state_dict(), best_model_path)
        
        # if len(val_loss_history) > 1:
        #     if val_loss - val_loss_history[-1] < 0.0001:
        #         break

        # Save the metrics
        val_loss_history.append(val_loss)
        val_balanced_acc_history.append(val_balanced_accuracy)
        val_roc_auc_history.append(val_roc)
        val_avg_precision_history.append(val_precision)

        if epoch == args.epochs-1:
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot()
        
    df = pd.DataFrame({
        'train_loss': train_loss_history,
        'train_balanced_accuracy': train_balanced_acc_history,
        'train_roc_auc': train_roc_auc_history,
        'train_avg_precision': train_avg_precision_history,
        'val_loss': val_loss_history,
        'val_balanced_accuracy': val_balanced_acc_history,
        'val_roc_auc': val_roc_auc_history,
        'val_avg_precision': val_avg_precision_history
    })

    # plot_summary_metrics(df, fold, "/home/user/persistent/KL Grade Prediction/plots")
   

def validation_model(model, val_loader, args):
    running_loss = 0.0
    # 1. set eval mode
    model.eval()
    model.to(device)

    y_pred  = []
    y_true = []
    y_pred_soft = []

    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        val_running_loss = 0.0
        for batch in val_loader:
            inputs = batch["img"].to(device)
            targets = batch["target"].to(device)
            
            outputs = model(inputs)
            a = outputs.detach().cpu().numpy()
            
            outputs_softmax = torch.softmax(outputs, dim=1)
            _,predicted = torch.max(outputs_softmax, 1)
                
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_pred_soft.extend(outputs_softmax.cpu().numpy())

            # # calc balanced accuracy
            # val_balanced_accuracy = balanced_accuracy_score(targets, predicted)
            # val_bal_accuracy += val_balanced_accuracy
            # # print(f"balanced_acc: {val_balanced_accuracy}")
            # # val_multicalss_accuracy = MulticlassAccuracy(num_classes=5)
            # # val_multicalss_accuracy.update(outputs_softmax, targets)
            # # val_bal_accuracy.append(val_multicalss_accuracy.compute().mean().item())
                

            # # calc roc-auc
            # val_rocauc = MulticlassAUROC(num_classes=5, average=None)
            # val_rocauc.update(outputs_softmax, targets)
            # val_roc.append(val_rocauc.compute().mean().item())
                
            # # calc avg precision    
            # val_avg_precision = MulticlassAUPRC(num_classes=5, average="macro")
            # val_avg_precision.update(outputs_softmax, targets)
            # val_prc.append(val_avg_precision.compute().item())

            # val_confusion_matrix = MulticlassConfusionMatrix(num_classes=5)
            # val_confusion_matrix.update(outputs_softmax, targets)
            # print(f"Confusion Matrix: {val_confusion_matrix.compute()}")
                
            # calc loss
            loss = criterion(outputs,targets)
            val_running_loss += loss.item()


        # precison = sum(val_prc) / len(val_prc)
        # roc = sum(val_roc) / len(val_roc)   
        # balanced_accuracy = val_balanced_accuracy / len(val_loader)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        roc = roc_auc_score(y_true, y_pred_soft, multi_class="ovo", average="macro")
        precison = average_precision_score(y_true, y_pred_soft, average="macro")
        loss  = val_running_loss / len(val_loader)      

        
        print(f"Validation Loss: {loss / len(val_loader)}")
    return loss, balanced_accuracy, roc, precison

def plot_summary_metrics(df, fold, out_dir):
    """
    Plots the training and validation metrics over all epochs after training completes.
    Saves the summary plot in the specified output directory.

    Args:
    - df (pd.DataFrame): DataFrame containing the training and validation metrics over epochs.
    - fold (int): The fold number for which metrics are being plotted.
    - out_dir (str): Directory where the summary plot will be saved.
    """

    # Set up subplots for all metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Summary of Training and Validation Metrics for Fold {fold}')

    # Plot training and validation loss
    axes[0, 0].plot(df['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(df['val_loss'], label='Validation Loss', color='orange')
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    # Plot balanced accuracy
    axes[0, 1].plot(df['train_balanced_accuracy'], label='Train Balanced Accuracy', color='blue')
    axes[0, 1].plot(df['val_balanced_accuracy'], label='Validation Balanced Accuracy', color='orange')
    axes[0, 1].set_title("Balanced Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Balanced Accuracy")
    axes[0, 1].legend()

    # Plot ROC-AUC score
    axes[1, 0].plot(df['train_roc_auc'], label='Train ROC-AUC', color='blue')
    axes[1, 0].plot(df['val_roc_auc'], label='Validation ROC-AUC', color='orange')
    axes[1, 0].set_title("ROC-AUC")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("ROC-AUC")
    axes[1, 0].legend()

    # Plot average precision
    axes[1, 1].plot(df['train_avg_precision'], label='Train Average Precision', color='blue')
    axes[1, 1].plot(df['val_avg_precision'], label='Validation Average Precision', color='orange')
    axes[1, 1].set_title("Average Precision")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Average Precision")
    axes[1, 1].legend()

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the summary plot
    plot_filename = os.path.join(out_dir, f"summary_metrics_fold_{fold}.png")
    plt.savefig(plot_filename)
    
    plt.show()
    plt.close(fig)
