import datetime

import numpy as np
import scipy.sparse
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats.mstats import gmean
tqdm.pandas(desc="my bar!")

import pandas as pd

from pathlib import Path
from scipy.sparse import csr_matrix, save_npz, vstack

num_classes = 5270


average_type = 'gmean'

# model_name = 'resnet50f_48'
# model_name = 'resnet101f_34'
model_name = 'resnet152f_22'

model_path = Path('data/prediction') / model_name

test_hashes = pd.read_csv('data/test_hashes.csv')

test_hashes['file_name'] = test_hashes['file_name'].str.split('/').str.get(-1)
print('test_hashes.shape = ', test_hashes.shape)

# test_hashes = test_hashes.drop_duplicates(subset=['image_id', 'md5'])
# print('test_hashes.shape = ', test_hashes.shape)

file_to_md5 = dict(zip(test_hashes['file_name'].values, test_hashes['md5'].values))

preds = sorted(list(model_path.glob('last_test_[0,1,2,3,4,5,6,7].npz')))

print('[{}] Loading predictions...'.format(str(datetime.datetime.now())))

ms = [scipy.sparse.load_npz(str(x)) for x in preds]

print('[{}] Loading test file_names...'.format(str(datetime.datetime.now())))

file_names = pd.read_csv(str(model_path / 'last_test_0.csv'))

# train_df = pd.read_csv('data/train1_df.csv')

# map_dict = dict(zip(train_df['class_id'].values, train_df['category_id'].values))

file_names['file_name'] = file_names['file_name'].str.split('/').str.get(-1)

print('[{}] Preparing md5 => prob...'.format(str(datetime.datetime.now())))


def get_probs(i, average_type='gmean'):
    image_name = file_names.loc[i, 'file_name']
    temp = []

    for j, m in enumerate(ms):
        temp += [m[i]]

    if average_type == 'mean':
        temp = scipy.sparse.vstack(temp).mean(axis=0)
    elif average_type == 'gmean':
        temp = gmean(scipy.sparse.vstack(temp).todense() + 1e-15, axis=0)

    temp[temp < 1e-6] = 0

    return file_to_md5[image_name], csr_matrix(temp)


result = Parallel(n_jobs=8)(delayed(get_probs)(i) for i in file_names.index)
#
# result = [get_probs(i) for i in tqdm(file_names.index)]

print('[{}] Unzippping...'.format(str(datetime.datetime.now())))

pred_md5_list, probs = zip(*result)

probs = vstack(probs)

labels = pd.DataFrame({'md5': pred_md5_list})

print('[{}] Saving labels...'.format(str(datetime.datetime.now())))

labels.to_csv(str(model_path / (average_type + '_last_md5_list.csv')), index=False)

print('[{}] Saving predictions...'.format(str(datetime.datetime.now())))

save_npz(str(model_path / (average_type + '_last_probs.npz')), probs)
