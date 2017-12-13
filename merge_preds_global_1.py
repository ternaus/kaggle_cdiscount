import datetime

import numpy as np
import scipy.sparse
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats.mstats import gmean

tqdm.pandas(desc="my bar!")

import pandas as pd
import json
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz, vstack

num_classes = 5270

average_type = 'gmean'

model_names = ['resnet101_23',
               'resnet101f_34',
               'resnet152_23',
               'resnet152f_22',
               'resnet50_36',
               'resnet50f_48']


config = json.loads(open(str(Path('__file__').absolute().parent / 'config.json')).read())

data_path = Path(config['data_dir']).expanduser()

source_file_name_nps = 'last_test_4.npz'

preds = [data_path / 'prediction' / model_name / source_file_name_nps for model_name in model_names]

test_hashes = pd.read_csv(str(data_path / 'test_hashes.csv'))

test_hashes['file_name'] = test_hashes['file_name'].str.split('/').str.get(-1)
print('test_hashes.shape = ', test_hashes.shape)

file_to_md5 = dict(zip(test_hashes['file_name'].values, test_hashes['md5'].values))

print('[{}] Loading predictions...'.format(str(datetime.datetime.now())))

ms = [scipy.sparse.load_npz(str(x)) for x in preds]

print('[{}] Loading test file_names...'.format(str(datetime.datetime.now())))

file_names = pd.read_csv(str(data_path / 'prediction' / model_names[0] / 'last_test_4.csv'))

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


result_path = Path('data') / 'prediction' / 'global'
result_path.mkdir(exist_ok=True, parents=True)

result = Parallel(n_jobs=12)(delayed(get_probs)(i) for i in file_names.index)
#
# result = [get_probs(i) for i in tqdm(file_names.index)]

print('[{}] Unzippping...'.format(str(datetime.datetime.now())))

pred_md5_list, probs = zip(*result)

probs = vstack(probs)

labels = pd.DataFrame({'md5': pred_md5_list})

print('[{}] Saving labels...'.format(str(datetime.datetime.now())))

labels.to_csv(str(result_path / (average_type + '_last_md5_list.csv')), index=False)

print('[{}] Saving predictions...'.format(str(datetime.datetime.now())))

save_npz(str(result_path / (average_type + '_last_probs.npz')), probs)
