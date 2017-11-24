from pathlib import Path
import cv2

import pandas as pd

from tqdm import tqdm

train_imgs = list(Path('../data/train').glob('**/*.jpg'))


def helper(file_name):
    img = cv2.imread(str(file_name))
    return img.var(axis=(0, 1))


result = [helper(x) for x in tqdm(train_imgs)]

df = pd.DataFrame(result, columns=[0, 1, 2])

file_names = [str(x) for x in tqdm(train_imgs)]

df.to_csv('../data/var.csv', index=False)
