import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold



main_dir = r"C:\Knee OE KL Grade Prediction\data\Images"
out_dir = r"C:\Knee OE KL Grade Prediction\data\CSV/data.csv"

# wohin die 5fold dateien
output_dir = r"C:\Knee OE KL Grade Prediction\data\CSV"
def create_csv_and_folds():
    meta_data = pd.DataFrame(columns=[
        "Name", "Path", "KL"
    ])


    for grade in range(5):
        folder_path = os.path.join(main_dir, f"{grade}")

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path,image_name)

            meta_data= meta_data._append({
                "Name": image_name,
                "Path": image_path,
                "KL": grade
            }, ignore_index=True)

    meta_data.to_csv(out_dir, index=False)


    #-------------- Train Test split ---------------#
    X = meta_data.drop(columns=["KL"], inplace=False)
    y = meta_data["KL"]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


    #-------------- Train Val split ---------------#
    X_train, X_valdiation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2) 

    # plots of the distribution
    y_train.value_counts().sort_index().plot(kind="bar")
    # plt.savefig("train_distribution")
    y_test.value_counts().sort_index().plot(kind="bar")
    # plt.savefig("test_distribution")
    y_validation.value_counts().sort_index().plot(kind="bar")
    # plt.savefig("validation_distribution")



    df = pd.read_csv(out_dir)
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
        # plt.savefig(os.path.join(plot_path, f"fold_{fold}_train_distribution"))

        val_data = pd.concat([X_test, y_test], axis=1)
        val_data["KL"].value_counts().plot(kind="bar")
        # plt.savefig(os.path.join(plot_path, f"fold_{fold}_val_distribution"))

        path_val = os.path.join(output_dir,f"fold_{fold}_val.csv")
        val_data.to_csv(path_val, index=False)

        fold += 1

if __name__ == '__main__':
    create_csv_and_folds()
