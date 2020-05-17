from pathlib import Path
import h5py
import os
import time
import numpy as np
import json
import argparse

# Add arguments to parser
parser = argparse.ArgumentParser(description='Generate MLM entities')
parser.add_argument('--dataset', default='MLM_v1_eu', type=str,
                        choices=['MLM_v1', 'MLM_v1_sample', 'MLM_v1_eu', 'MLM_v2'], help='dataset')
args = parser.parse_args()

ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent

train = h5py.File(os.path.join(ROOT_PATH, f'dataset/{args.dataset}/train.h5'), 'r')
test = h5py.File(os.path.join(ROOT_PATH, f'dataset/{args.dataset}/test.h5'), 'r')
val = h5py.File(os.path.join(ROOT_PATH, f'dataset/{args.dataset}/val.h5'), 'r')

all_hdf5 = {
    'train': train,
    'val': val,
    'test': test
}

all_ids = [('train', list(train['ids'])), ('val', list(val['ids'])), ('test', list(test['ids']))]

def save_file(fileName, file):
    with open(fileName, 'w') as outfile:
        json.dump(file, outfile, ensure_ascii=False, indent=4)

clusters = {cell: [] for cell in range(0, len(list(train[f'{list(all_ids[0][1])[0]}_onehot'][()])))}

tic = time.perf_counter()
for part, ids in all_ids:
    hdf5 = all_hdf5[part]
    for i, id in enumerate(ids):
        onehot = hdf5[f'{id}_onehot'][()]
        clusters[np.argmax(onehot)].append(int(id))
        toc = time.perf_counter()
        print(f'====> Finished id {id} -- {((i+1)/len(ids))*100:.2f}% -- {toc - tic:0.2f}s -- {part}')
    save_file(f'clusters_{part}', clusters)
    for value in clusters.values(): del value[:] # clear dict values for next part

print('Done!')