import numpy as np
import pandas as pd
import torch.utils.data as data
import torch
import utils

num_classes = 5270


class CSVDataset(data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.path = df['file_name'].values.astype(str)
        self.target = df['class_id'].values.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        X = utils.load_image(self.path[idx])
        if self.transform:
            X = self.transform(X)
        y = self.target[idx]
        return X, y


def get_loaders(batch_size,
                args,
                train_transform=None,
                valid_transform=None):
    train_df = pd.read_csv(f'data/train4_df.csv')

    train_dataset = CSVDataset(train_df, transform=train_transform)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=args.workers,
                                   pin_memory=torch.cuda.is_available())

    valid_df = pd.read_csv(f'data/val4_df.csv')

    valid_dataset = CSVDataset(valid_df, transform=valid_transform)
    valid_loader = data.DataLoader(valid_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=torch.cuda.is_available())

    return train_loader, valid_loader
