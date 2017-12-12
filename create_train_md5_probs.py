"""
dumps dictionary

md5 => probs

for train md5's that are in test
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import datetime

num_classes = 5270

print('[{}] Reading dataframes...'.format(str(datetime.datetime.now())))

train_hashes = pd.read_csv('data/train_hashes.csv')
test_hashes = pd.read_csv('data/test_hashes.csv')

train_df = pd.read_csv('data/train1_df.csv')

df_for_map = train_df[['category_id', 'class_id']].drop_duplicates()
df_for_map.to_csv('data/map_dict.csv', index=False)

mapping_dict = dict(zip(df_for_map['category_id'].astype(str).values, df_for_map['class_id'].values))

result_md5s = set(train_hashes['md5'].unique()).intersection(set(test_hashes['md5'].unique()))

print('Num md5 in result = ', len(result_md5s))


hashes = train_hashes[train_hashes['md5'].isin(result_md5s)]

hashes['category_id'] = hashes['file_name'].str.split('/').str.get(-2)

hashes['class_id'] = hashes['category_id'].map(mapping_dict)

g = hashes.groupby('md5')

md5_map = {}

for md5, df in tqdm(g):
    if df['class_id'].nunique() == 1:
        tt = [int(df.iloc[0]['class_id'])]
    else:
        tt = list(df['class_id'].values.astype(int))

    md5_map[md5] = tt

expanded_map = {}

for key, value in tqdm(md5_map.items()):
    temp = np.zeros((len(value), num_classes))
    for ind, class_ind in enumerate(value):
        temp[(ind, class_ind)] = 1

    expanded_map[key] = csr_matrix(temp.mean(axis=0))

print('Num md5 in result')
print(len(expanded_map))

pickle_out = open("data/test_md5_map.pickle", "wb")
pickle.dump(expanded_map, pickle_out)
pickle_out.close()
