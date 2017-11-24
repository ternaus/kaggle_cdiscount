"""
Find md5's
"""
import hashlib
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed


def get_md5(file_name):
    return hashlib.md5(open(str(file_name), 'rb').read()).hexdigest()


if __name__ == '__main__':
    test_path = Path('../data') / 'test'

    test_file_names = list(test_path.glob('**/*.jpg'))

    hashes = Parallel(n_jobs=100)(delayed(get_md5)(file_name) for file_name in test_file_names)

    df = pd.DataFrame({'file_name': test_file_names,
                       'md5': hashes})

    df.to_csv('../data/test_hashes.csv', index=False)
