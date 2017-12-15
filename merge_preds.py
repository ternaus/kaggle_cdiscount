import datetime
import json

import scipy.sparse
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats.mstats import gmean

tqdm.pandas(desc="my bar!")

import pandas as pd

from pathlib import Path
from scipy.sparse import csr_matrix, save_npz, vstack
import argparse
import gc


def get_hashes(filtered_hashes=False):
    test_hashes = pd.read_csv(str(data_path / 'test_hashes.csv'))

    test_hashes['file_name'] = test_hashes['file_name'].str.split('/').str.get(-1)

    if not filtered_hashes:
        return test_hashes
    else:
        train_hashes = pd.read_csv(str(data_path / 'train_hashes.csv'))
        test_hashes = test_hashes.drop_duplicates('md5')
        test_hashes = test_hashes[~test_hashes['md5'].isin(set(train_hashes['md5'].unique()))]
    return test_hashes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg = parser.add_argument

    arg('--model_type', type=str, default='last', help='what model to use last or best')
    arg('--workers', type=int, default=12)
    arg('--model_name', type=str, help='can be '
                                       'last '
                                       'mean '
                                       'all ')

    arg('--average_type', type=str, default='gmean')
    args = parser.parse_args()

    print('[{}] Setting up paths...'.format(str(datetime.datetime.now())))

    config = json.loads(open(str(Path('__file__').absolute().parent / 'config.json')).read())
    data_path = Path(config['data_dir']).expanduser()
    model_path = data_path / 'prediction' / args.model_name

    print('[{}] Getting hashes...'.format(str(datetime.datetime.now())))

    hashes = get_hashes()

    file2md5 = dict(zip(hashes['file_name'].values, hashes['md5'].values))

    if args.model_type != 'all':
        tta_file_names = list(model_path.glob('{model_type}*.csv'.format(model_type=args.model_type)))
    else:
        tta_file_names = list(model_path.glob('*.csv'))

    print('Averaging {num_tta} for {model_name}'.format(num_tta=len(tta_file_names), model_name=args.model_name))

    md5_2_ind_dicts = []

    common_md5s = None

    for file_name in tqdm(tta_file_names):
        df = pd.read_csv(str(file_name))

        df['file_name'] = df['file_name'].str.split('/').str.get(-1)

        df['md5'] = df['file_name'].map(file2md5)

        md5_2_ind = dict(zip(df['md5'].values, range(df.shape[0])))

        md5_2_ind_dicts += [md5_2_ind]

        if not common_md5s:
            common_md5s = set(md5_2_ind.keys())
        else:
            common_md5s = common_md5s.intersection(set(md5_2_ind.keys()))

    print('number of common md5s =  {num_file_names}'.format(num_file_names=len(common_md5s)))

    print('[{}] Reading  important hashes...'.format(str(datetime.datetime.now())))

    important_hashes = get_hashes(filtered_hashes=True)
    common_md5s = sorted(list(common_md5s.intersection(set(important_hashes['md5'].values))))

    print('[{}] creating merged dict...'.format(str(datetime.datetime.now())))
    print('number of common md5s =  {num_file_names}'.format(num_file_names=len(common_md5s)))

    md5_2_ind_joined = {}

    for md5 in tqdm(common_md5s):
        temp = []
        for md5_2_ind in md5_2_ind_dicts:
            temp += [md5_2_ind[md5]]

        md5_2_ind_joined[md5] = temp


    def get_probs(md5, average_type='gmean'):
        temp = []

        for position, row in enumerate(md5_2_ind_joined[md5]):
            temp += [ms[position][row]]

        if average_type == 'mean':
            temp = scipy.sparse.vstack(temp).mean(axis=0)
        elif average_type == 'gmean':
            temp = gmean(scipy.sparse.vstack(temp).todense() + 1e-15, axis=0)

        temp[temp < 1e-6] = 0

        return md5, csr_matrix(temp)

    gc.collect()

    print('[{}] Loading predictions...'.format(str(datetime.datetime.now())))
    ms = [scipy.sparse.load_npz(str(x).replace('csv', 'npz')) for x in tqdm(tta_file_names)]

    result = Parallel(n_jobs=12)(delayed(get_probs)(md5) for md5 in common_md5s)

    # result = [get_probs(i) for i in tqdm(file_names.index)]

    print('[{}] Unzippping...'.format(str(datetime.datetime.now())))

    pred_md5_list, probs = zip(*result)

    probs = vstack(probs)

    labels = pd.DataFrame({'md5': pred_md5_list})

    print('[{}] Saving labels...'.format(str(datetime.datetime.now())))

    labels.to_csv(str(model_path / (args.average_type + '_{model_type}_md5_list.csv'.format(model_type=args.model_type))), index=False)

    print('[{}] Saving predictions...'.format(str(datetime.datetime.now())))

    save_npz(str(model_path / (args.average_type + '_{model_type}_probs.npz'.format(model_type=args.model_type))), probs)
