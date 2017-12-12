import bson
import pandas as pd
from pathlib import Path


import multiprocessing as mp

prod_to_category = mp.Manager().dict()  # note the difference


def process(q, iolock):
    while True:
        d = q.get()
        if d is None:
            break
        p_id = d['_id']

        for e, pic in enumerate(d['imgs']):
            fname = str(base_path / '{p_id}_{e}.jpg'.format(p_id=p_id, e=e))

            with open(fname, 'wb') as f:
                f.write(pic['picture'])


data_path = Path('data')
base_path = data_path / 'test'
base_path.mkdir(exist_ok=True)


n_cores = 12
prods = mp.Manager().dict()

q = mp.Queue(maxsize=n_cores)
iolock = mp.Lock()
pool = mp.Pool(n_cores, initializer=process, initargs=(q, iolock))


# process the file

data = bson.decode_file_iter(open(str(data_path / 'test.bson'), 'rb'))
for c, d in enumerate(data):
    q.put(d)

# tell workers we're done

for _ in range(n_cores):
    q.put(None)

pool.close()
pool.join()

# convert back to normal dictionary
prod_to_category = dict(prod_to_category)

prod_to_category = pd.DataFrame.from_dict(prod_to_category, orient='index')
prod_to_category.index.name = '_id'
prod_to_category.rename(columns={0: 'category_id'}, inplace=True)