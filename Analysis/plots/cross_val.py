import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt


# paths 
output_dir = r"C:\Knee OE KL Grade Prediction\data\CSV"
plot_path = r"C:\Knee OE KL Grade Prediction\Analysis\plots"


df = pd.read_csv(r"C:\Knee OE KL Grade Prediction\data\CSV/data.csv")
train = df.drop(columns=["KL"])
test = df["KL"]

# split data in train and test
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.2)

# save test set to dir
test_data = pd.concat([X_test, y_test], axis=1)
path_train = os.path.join(output_dir, "test_data.csv")
test_data.to_csv(path_train ,index=False)


X = X_train
y = y_train
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# iterate over the stratified splits
fold = 0
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

 
    train_data = pd.concat([X_train, y_train], axis=1)
    path_train = os.path.join(output_dir, f"fold_{fold}_train.csv")
    train_data.to_csv(path_train ,index=False)

    train_data["KL"].value_counts().plot(kind="bar")
    plt.savefig(os.path.join(plot_path, f"fold_{fold}_train_distribution"))

    val_data = pd.concat([X_test, y_test], axis=1)
    val_data["KL"].value_counts().plot(kind="bar")
    plt.savefig(os.path.join(plot_path, f"fold_{fold}_val_distribution"))

    path_val = os.path.join(output_dir,f"fold_{fold}_val.csv")
    val_data.to_csv(path_val, index=False)

    fold += 1
