from pathlib import Path

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd

train_path = Path('../data/train')

file_names = list(map(str, train_path.glob('**/*.jpg')))

df = pd.DataFrame({'file_name': file_names})

df['category_id'] = df['file_name'].str.split('/').str.get(3)

le = LabelEncoder()

df['class_id'] = le.fit_transform(df['category_id'])

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=2016)

for train_index, val_index in sss.split(df['class_id'], df['class_id']):
    df_train = df.iloc[train_index]
    print(df_train.shape)
    df_train.to_csv('../data/train_df.csv', index=False)

    df_val = df.iloc[val_index]
    print(df_val.shape)
    df_val.to_csv('../data/val_df.csv', index=False)
    break
