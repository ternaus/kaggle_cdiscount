from pathlib import Path
import cv2
from joblib import Parallel, delayed
import pandas as pd

from tqdm import tqdm

test_imgs = list(Path('../data/test').glob('**/*.jpg'))


def helper(file_name):
    img = cv2.imread(str(file_name))
    return img.var(axis=(0, 1))


# result = [helper(x) for x in tqdm(train_imgs)]
result = Parallel(n_jobs=8)(delayed(helper)(x) for x in test_imgs)

df = pd.DataFrame(result, columns=[0, 1, 2])

file_names = [str(x) for x in tqdm(test_imgs)]

df['file_names'] = file_names

df.to_csv('../data/var_test.csv', index=False)
