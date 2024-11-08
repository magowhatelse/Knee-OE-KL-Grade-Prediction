import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Knee OA Classification')
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'resnet34', 'resnet50'],
                                                                                         default='resnet18')
    parser.add_argument('--out_dir', '--out_dir', type=str, default='session')

    parser.add_argument('--csv_dir', '--csv_dir', default='data/CSV')

    parser.add_argument('--batch_size', '--batch_size', default=128, type=int,
                                            choices=[16, 32, 64, 128])
    parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float,
                        choices=[0.0001, 0.00001])

    parser.add_argument('--epochs', '--epochs', default=100, type=int)

    args = parser.parse_args()

    return args