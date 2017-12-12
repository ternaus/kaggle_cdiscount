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

import models
import utils


class PredictionDataset:
    def __init__(self, paths, aug: int, transform):
        self.paths = paths
        self.aug = aug
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        image = utils.load_image(path)

        if self.aug == 0:
            image = image.crop((0, 0, 160, 160))
        elif self.aug == 1:
            image = image.crop((20, 0, 160, 160))
        elif self.aug == 2:
            image = image.crop((0, 20, 160, 160))
        elif self.aug == 3:
            image = image.crop((20, 20, 160, 160))
        elif self.aug == 4:
            image = image.crop((10, 10, 160, 160))
        elif self.aug == 5:
            image = image.crop((10, 0, 160, 160))
        elif self.aug == 6:
            image = image.crop((0, 10, 160, 160))
        elif self.aug == 7:
            image = image.crop((20, 10, 160, 160))
        elif self.aug == 8:
            image = image.crop((10, 20, 160, 160))
        else:
            raise Exception('Wrong augmentation')

        return self.transform(image), str(path)


def predict(model, paths, batch_size: int, aug=False, transform=None):
    loader = DataLoader(
        dataset=PredictionDataset(paths, aug, transform),
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
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


def get_model(model_name, num_classes, device_ids):

    device_ids = list(map(int, device_ids.split(',')))

    if 'resnet101' in model_name:
        model = models.ResNetFinetune(num_classes, net_cls=models.M.resnet101)
    elif 'resnet50' in model_name:
        model = models.ResNetFinetune(num_classes, net_cls=models.M.resnet50)
    elif 'resnet152' in model_name:
        model = models.ResNetFinetune(num_classes, net_cls=models.M.resnet152)

    model = nn.DataParallel(model, device_ids=device_ids).cuda()

    if args.model_type == 'best':
        state = torch.load(str(Path('models') / model_name / 'best-model.pt'.format(model_name=model_name)))
    else:
        state = torch.load(str(Path('models') / model_name / 'model.pt'.format(model_name=model_name)))

    model.load_state_dict(state['model'])

    return model


def add_args(parser):
    arg = parser.add_argument
    arg('--batch-size', type=int, default=128)
    arg('--model_type', type=str, default='best', help='what model to use last or best')
    arg('--workers', type=int, default=8)
    arg('--model', type=str)
    arg('--mode', type=str, default='val', help='can be test or val')
    arg('--aug', type=int, default='4', help='0, 1, 2, 3, 4, 5, 6, 7, 8')
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

    model_name = 'resnet152f_22'

    model = get_model(model_name, num_classes, args.device_ids)

    model_path = data_path / 'prediction' / model_name
    model_path.mkdir(exist_ok=True, parents=True)

    target_size = 160

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    aug = args.aug

    print(aug)

    if args.mode == 'val':
        val_df = pd.read_csv(str(data_path / 'val4_df.csv'))
        preds, labels = predict(model, val_df['file_name'].apply(Path).values, batch_size, aug=aug)
        target_file_name = args.model_type + '_test_' + str(aug)

    elif args.mode == 'test':
        test_hashes = pd.read_csv(str(data_path / 'test_hashes.csv'))
        # train_hashes = pd.read_csv(str(data_path / 'train_hashes.csv'))
        # test_hashes = test_hashes.drop_duplicates('md5')
        # test_hashes = test_hashes[~test_hashes['md5'].isin(set(train_hashes['md5'].unique()))]
        # bad_md5 = ['d704b9555801285eedb04213a02fdc41', '35e7e038fe2ec215f63bdb5e4b739524']
        # test_hashes = test_hashes[~test_hashes['md5'].isin(set(bad_md5))]

        preds, labels = predict(model, test_hashes['file_name'].apply(Path).values, batch_size, aug=aug,
                                transform=transform)

        target_file_name = args.model_type + '_test_' + str(aug)

    labels = pd.DataFrame(labels, columns=['file_name'])

    print('[{}] Saving labels...'.format(str(datetime.datetime.now())))

    labels.to_csv(str(model_path / (target_file_name + '.csv')), index=False)

    print('[{}] Saving predictions...'.format(str(datetime.datetime.now())))

    save_npz(str(model_path / (target_file_name + '.npz')), preds)
