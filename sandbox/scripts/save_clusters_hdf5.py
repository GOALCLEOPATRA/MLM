import os
import h5py
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import argparse

def read_json(path):
    with open(path) as json_data:
        return json.load(json_data)

# Add arguments to parser
parser = argparse.ArgumentParser(description='Generate MLM entities')
parser.add_argument('--dataset', default='MLM_v1_eu', type=str,
                        choices=['MLM_v1', 'MLM_v1_sample', 'MLM_v1_eu', 'MLM_v2'], help='dataset')
args = parser.parse_args()

# define paths
ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent

train = h5py.File(os.path.join(ROOT_PATH, f'dataset/{args.dataset}/train.h5'), 'a')
test = h5py.File(os.path.join(ROOT_PATH, f'dataset/{args.dataset}/test.h5'), 'a')
val = h5py.File(os.path.join(ROOT_PATH, f'dataset/{args.dataset}/val.h5'), 'a')

all_hdf5 = {
    'train': [train, read_json(ROOT_PATH.parent / 'clusters_train')],
    'val': [val, read_json(ROOT_PATH.parent / 'clusters_val')],
    'test': [test, read_json(ROOT_PATH.parent / 'clusters_test')]
}

all_ids = [('train', train['ids']), ('val', val['ids']), ('test', test['ids'])]

tic = time.perf_counter()
for part, ids in all_ids:
    # read hdf5 and clusters
    h5f = all_hdf5[part][0]
    clusters = all_hdf5[part][1]
    for i, id in enumerate(ids):
        onehot = h5f[f'{id}_onehot'][()]
        cell = str(np.argmax(onehot))
        cluster = set(clusters[cell]).copy() # get cluster
        cluster.remove(id) # remove id from cluster
        if f'{id}_cluster' not in h5f:
            h5f.create_dataset(name=f'{id}_cluster', data=np.array(list(cluster), dtype=np.int), compression="gzip", compression_opts=9)

        toc = time.perf_counter()
        print(f'====> Finished id {id} -- {((i + 1) / len(ids)) * 100:.2f}% -- {toc - tic:0.2f}s -- {part}')

    # close hdf5 files
    h5f.close()
