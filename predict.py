"""
Script generates predictions from model.
"""

import argparse
import datetime
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, save_npz, vstack
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


import utils
import models


class PredictionDatasetPure:
    def __init__(self, paths, n_test_aug):
        self.paths = paths
        self.n_test_aug = n_test_aug

    def __len__(self):
        return len(self.paths) * self.n_test_aug

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        image = utils.load_image(path)
        return transform_pure(image), str(path)


class PredictionDatasetAug:
    def __init__(self, paths, n_test_aug):
        self.paths = paths
        self.n_test_aug = n_test_aug

    def __len__(self):
        return len(self.paths) * self.n_test_aug

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        image = utils.load_image(path)
        return transform_aug(image), str(path)


def predict(model, paths, batch_size: int, n_test_aug: int, aug=False):
    if aug:
        loader = DataLoader(
            dataset=PredictionDatasetAug(paths, n_test_aug),
            shuffle=False,
            batch_size=batch_size,
            num_workers=args.workers,
            pin_memory=True
        )
    else:
        loader = DataLoader(
            dataset=PredictionDatasetPure(paths, n_test_aug),
            shuffle=False,
            batch_size=batch_size,
            num_workers=args.workers,
            pin_memory=True
        )
    threshold = 1e-5  # will cut off 99% of the data

    model.eval()
    all_outputs = []
    all_stems = []
    for inputs, stems in tqdm(loader, desc='Predict'):
        inputs = utils.variable(inputs, volatile=True)
        outputs = F.softmax(model(inputs))

        outputs[outputs < threshold] = 0

        sparse_outputs = csr_matrix(outputs.data.cpu().numpy())

        all_outputs.append(sparse_outputs)
        all_stems.extend(stems)

    return vstack(all_outputs), all_stems


def get_model(model_name, num_classes):
    if 'resnet101' in model_name:
        model = models.ResNetFinetune(num_classes, net_cls=models.M.resnet101)

    model = nn.DataParallel(model, device_ids=[0]).cuda()

    if args.model_type == 'best':
        state = torch.load(str(Path('models') / model_name / 'best-model.pt'.format(model_name=model_name)))
    else:
        state = torch.load(str(Path('models') / model_name / 'model.pt'.format(model_name=model_name)))

    model.load_state_dict(state['model'])

    return model


def add_args(parser):
    arg = parser.add_argument
    arg('--batch-size', type=int, default=256)
    arg('--model_type', type=str, default='best', help='what model to use last or best')
    arg('--workers', type=int, default=12)
    arg('--model', type=str)
    arg('--mode', type=str, default='val', help='can be test or val')
    arg('--aug', type=str, default='center', help='Type of the augmentation, random or center')
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    add_args(parser)
    args = parser.parse_args()

    timestamp = int(time.time())

    batch_size = args.batch_size
    num_classes = 5270

    data_path = Path('data')

    model_name = 'resnet101_18'
    model = get_model(model_name, num_classes)

    model_path = data_path / 'prediction' / model_name
    model_path.mkdir(exist_ok=True, parents=True)

    target_size = 160

    transform_aug = transforms.Compose([
        transforms.RandomCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_pure = transforms.Compose([
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    aug = args.aug != 'center'

    print(aug)

    if args.mode == 'val':
        val_df = pd.read_csv(str(data_path / 'val3_df.csv'))
        preds, labels = predict(model, val_df['file_name'].apply(Path), batch_size, 1, aug=aug)
        target_file_name = args.model_type + '_val_' + str(timestamp)

    elif args.mode == 'test':
        test_path = data_path / 'test'
        file_names = sorted(list(test_path.glob('*')))
        preds, labels = predict(model, file_names, batch_size, 1, aug=aug)
        target_file_name = args.model_type + '_test_' + str(timestamp)

    labels = pd.DataFrame(labels, columns=['file_name'])

    print('[{}] Saving labels...'.format(str(datetime.datetime.now())))

    labels.to_csv(str(model_path / (target_file_name + '.csv')), index=False)

    print('[{}] Saving predictions...'.format(str(datetime.datetime.now())))

    save_npz(str(model_path / (target_file_name + '.npz')), preds)
