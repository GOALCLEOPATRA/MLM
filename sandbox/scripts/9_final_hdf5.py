'''
Create one HDF5 file and store whole dataset
'''
import os
import h5py
import time
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add arguments to parser
parser = argparse.ArgumentParser(description='Generate MLM entities')
parser.add_argument('--dataset', default='MLM_v1_eu', type=str,
                        choices=['MLM_v1', 'MLM_v1_sample', 'MLM_v1_eu', 'MLM_v2'], help='dataset')
args = parser.parse_args()

# define paths
ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent
HDF5_DIR = ROOT_PATH.parent / f'mlm_dataset/hdf5/{args.dataset}'

# read data
images = h5py.File(os.path.join(HDF5_DIR, 'images/images.h5'), 'r') # 219123
summaries_bert = h5py.File(os.path.join(HDF5_DIR, 'summaries/summaries_bert.h5'), 'r')
coordinates = h5py.File(os.path.join(HDF5_DIR, 'coordinates/coordinates.h5'), 'r') # 218685
classes = h5py.File(os.path.join(HDF5_DIR, 'classes/classes_bert.h5'), 'r')

ids = list(coordinates['ids'])

# split to train, test and val
# 80% training, 10% validation and 10% test
train_ids, val_ids = train_test_split(ids, test_size=0.2, shuffle=True)
val_ids, test_ids = train_test_split(val_ids, test_size=0.5, shuffle=False)

# MLM
MLM_train = h5py.File(os.path.join(ROOT_PATH, f'dataset/{args.dataset}/train.h5'), 'w')
MLM_val = h5py.File(os.path.join(ROOT_PATH, f'dataset/{args.dataset}/val.h5'), 'w')
MLM_test = h5py.File(os.path.join(ROOT_PATH, f'dataset/{args.dataset}/test.h5'), 'w')

def save_hdf5(h5f, id, im, sum, coord, one_hot, cl):
    h5f.create_dataset(name=f'{id}_images', data=im, compression="gzip", compression_opts=9)
    h5f.create_dataset(name=f'{id}_summaries', data=sum, compression="gzip", compression_opts=9)
    h5f.create_dataset(name=f'{id}_coordinates', data=coord, compression="gzip", compression_opts=9)
    h5f.create_dataset(name=f'{id}_onehot', data=one_hot, compression="gzip", compression_opts=9)
    h5f.create_dataset(name=f'{id}_classes', data=cl, compression="gzip", compression_opts=9)

all_hdf5 = {
    'train': [MLM_train],
    'val': [MLM_val],
    'test': [MLM_test]
}

all_ids = [('train', train_ids), ('val', val_ids), ('test', test_ids)]

# keys = [k for k in images.keys()]
tic = time.perf_counter()
for id_type in all_ids:
    ids_save = []
    for i, id in enumerate(id_type[1]):
        if f'{id}_summaries' in all_hdf5[id_type[0]][0]:
            continue
        if f'{id}_images' not in images:
            continue
        img = images[f'{id}_images'][()]
        sum_bert = []
        if f'{id}_en' in summaries_bert:
            sum_bert.append(summaries_bert[f'{id}_en'][()])
        if f'{id}_de' in summaries_bert:
            sum_bert.append(summaries_bert[f'{id}_de'][()])
        if f'{id}_fr' in summaries_bert:
            sum_bert.append(summaries_bert[f'{id}_fr'][()])
        coord = coordinates[f'{id}_coordinates'][()]
        coord_onehot = coordinates[f'{id}_onehot'][()]
        cl = classes[f'{id}_classes'][()]
        if not sum_bert:
            continue

        # save values
        save_hdf5(all_hdf5[id_type[0]][0], id, img, np.stack(sum_bert, axis=0), coord, coord_onehot, np.squeeze(cl, axis=0))

        ids_save.append(id)
        toc = time.perf_counter()
        print(f'====> Finished id {id} -- {((i+1)/len(id_type[1]))*100:.2f}% -- {toc - tic:0.2f}s -- {id_type[0]}')

    # save keys
    all_hdf5[id_type[0]][0].create_dataset(name=f'ids', data=np.array(ids_save, dtype=np.int), compression="gzip", compression_opts=9)

    # close writing hdf5 files
    all_hdf5[id_type[0]][0].close()

# close reading hdf5 files
images.close()
summaries_bert.close()
coordinates.close()
classes.close()