
import numpy as np
import pandas as pd
from PIL import Image

import torch.utils.data as data
import torchvision.transforms as transforms
from pathlib import Path


num_classes = 5270


class CSVDataset(data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.path = df['file_name'].values.astype(str)
        self.target = df['class_id'].values.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    @staticmethod
    def _load_pil(path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __getitem__(self, idx):
        X = self._load_pil(self.path[idx])
        if self.transform:
            X = self.transform(X)
        y = self.target[idx]
        return X, y


def get_loaders(batch_size,
                args,
                train_transform=None,
                valid_transform=None):

    train_df = pd.read_csv(f'../data/train_df.csv')

    train_dataset = CSVDataset(train_df, transform=train_transform)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=args.workers,
                                   pin_memory=True)

    if not valid_transform:
        valid_transform = transforms.Compose([
          transforms.Scale(224),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    valid_df = pd.read_csv(f'../data/val_df.csv')

    valid_dataset = CSVDataset(valid_df, transform=valid_transform)
    valid_loader = data.DataLoader(valid_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=True)

    return train_loader, valid_loader


def load_image(path: Path):
    return Image.open(str(path)).convert('RGB')
