import argparse

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--backbone",
                        type=str,
                        choices=["se_resnet50","resnet18"],
                        default="resnet18")
    # define output directory (model results, weights)
    parser.add_argument("--out", "--out_dir", type=str, default="session") 
    # CSV 
    parser.add_argument("csv", "--csv_dir", default="data/CSV")

    parser.add_argument("vs", "--batch_size", type=int, default=16,
                        choices=[16, 32, 64])

    args = parser.parse_args()

    return args
