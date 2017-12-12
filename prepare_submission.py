import datetime
import pickle

import numpy as np
import scipy.sparse
from joblib import Parallel, delayed
from tqdm import tqdm

tqdm.pandas(desc="my bar!")

import pandas as pd

from pathlib import Path

num_classes = 5270

model_name = 'resnet50f_48'

pickle_in = open("data/test_md5_map.pickle", 'rb')
md5_map = pickle.load(pickle_in)
pickle_in.close()

print('size of expanded map - ', len(md5_map))  # md5 => prob

model_path = Path('data/prediction') / model_name

test_hashes = pd.read_csv('data/test_hashes.csv')

test_hashes['file_name'] = test_hashes['file_name'].str.split('/').str.get(-1)
test_hashes['image_id'] = test_hashes['file_name'].str.split('_').str.get(0)
print('test_hashes.shape = ', test_hashes.shape)

test_hashes = test_hashes.drop_duplicates(subset=['image_id', 'md5'])
print('test_hashes.shape = ', test_hashes.shape)


file_to_md5 = dict(zip(test_hashes['file_name'].values, test_hashes['md5'].values))
md5_to_file = dict(zip(test_hashes['md5'].values, test_hashes['file_name'].values))

preds = sorted(list(model_path.glob('last_test_[0,1,2,3].npz')))

print('[{}] Loading predictions...'.format(str(datetime.datetime.now())))
#
# ms = [scipy.sparse.load_npz(str(x)) for x in preds]

print('[{}] Loading test file_names...'.format(str(datetime.datetime.now())))

file_names = pd.read_csv(str(model_path / 'last_test_0.csv'))

train_df = pd.read_csv('data/train1_df.csv')

map_dict = dict(zip(train_df['class_id'].values, train_df['category_id'].values))

file_names['file_name'] = file_names['file_name'].str.split('/').str.get(-1)

print('[{}] Preparing md5 => prob...'.format(str(datetime.datetime.now())))


def get_probs(i):
    image_name = file_names.loc[i, 'file_name']
    temp = []

    for j, m in enumerate(ms):
        temp += [m[i]]

    temp = scipy.sparse.vstack(temp).mean(axis=0)

    return file_to_md5[image_name], scipy.sparse.csr_matrix(temp)
#
#
# def get_probs(i):
#     image_name = file_names.loc[i, 'file_name']
#     temp = np.zeros((len(ms), num_classes))
#
#     for j, m in enumerate(ms):
#         temp[j] = m[i].todense()
#
#     return file_to_md5[image_name], scipy.sparse.csr_matrix(temp.mean(axis=0))


result = Parallel(n_jobs=8)(delayed(get_probs)(i) for i in file_names.index)

print('Dict from list =', len(result))

result = dict(result)

print('size of preds =', len(result))  # should be the same as len(pred)

for md5 in md5_map:
    result[md5] = md5_map[md5]

print('size of preds with md5 =', len(result))  # should be the same as len(pred)
print(test_hashes['md5'].nunique())


print('[{}] filling prob...'.format(str(datetime.datetime.now())))

test_hashes['prob'] = test_hashes.map(result)

print(test_hashes['prob'].isnull().sum())
#
# temp = test_hashes.loc[test_hashes['prob'].notnull(), ['md5', 'prob']]
#
# md4_map_fill = dict(zip(temp['md5'].values, temp['prob'].values))
#
# test_hashes['prob'] = test_hashes['md5'].map(md4_map_fill)
#
# print(test_hashes['prob'].isnull().sum())
#
# print('[{}] Saving...'.format(str(datetime.datetime.now())))
#
# test_hashes.to_csv('data/1.csv', index=False)
