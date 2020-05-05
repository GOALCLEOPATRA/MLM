import os
import h5py
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from args import get_parser
import json

def open_json(fileName):
    try:
        with open(fileName) as json_data:
            d = json.load(json_data)
    except Exception as s:
        d=s
        print(d)
    return d


parser = get_parser()
args = parser.parse_args()

# define paths
ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent
# HDF5_DIR = ROOT_PATH.parent / 'mlm_dataset/hdf5'
HDF5_DIR = f'{ROOT_PATH}/{args.data_path}'


#  these come from cluster_coord_cells.py
cluster_train = open_json(ROOT_PATH / 'sandbox/scripts/clusters_train')
cluster_test = open_json(ROOT_PATH / 'sandbox/scripts/clusters_test')
cluster_val = open_json(ROOT_PATH / 'sandbox/scripts/clusters_val')

# images_loc_scene_2 = h5py.File('C:/Users/TahmasebzadehG/Desktop/n/images_v2_loc_scene_2.h5', 'r')
# kes = images_loc.keys()
partition = 'train'
# data_path = 'C:/users/TahmasebzadehG/PycharmProjects/MLM_3/dataset/bert'
train = h5py.File(f'{ROOT_PATH}/data/train.h5', 'r')
test = h5py.File(f'{ROOT_PATH}/data/test.h5', 'r')
val = h5py.File(f'{ROOT_PATH}/data/val.h5', 'r')
ks = list(train.keys())
ks = [k.split('_')[len(k.split('_'))-1] for k in ks]
kss = set(ks)
for k in ks:
    print(k)
train_w = h5py.File(f'{ROOT_PATH}/data/train_new.h5', 'w')
test_w = h5py.File(f'{ROOT_PATH}/data/test_new.h5', 'w')
val_w = h5py.File(f'{ROOT_PATH}/data/val_new.h5', 'w')

all_hdf5 = {
    'train': [train_w],
    'val': [val_w],
    'test': [test_w]
}

all_ids = [('train', train['ids']), ('val', val['ids']), ('test', test['ids'])]



def save_hdf5(h5f, id, im, sum, one_hot, coord, classes, clusters):
    h5f.create_dataset(name=f'{id}_images', data=im, compression="gzip", compression_opts=9)
    h5f.create_dataset(name=f'{id}_summaries', data=sum, compression="gzip", compression_opts=9)
    h5f.create_dataset(name=f'{id}_onehot', data=one_hot, compression="gzip", compression_opts=9)
    h5f.create_dataset(name=f'{id}_coordinates', data=coord, compression="gzip", compression_opts=9)
    h5f.create_dataset(name=f'{id}_classes', data=classes, compression="gzip", compression_opts=9)
    h5f.create_dataset(name=f'{id}_cluster', data=clusters, compression="gzip", compression_opts=9)


tic = time.perf_counter()

for id_type in all_ids:

    for i, id in enumerate(id_type[1]):


        if id_type[0] == 'train':
            coord_onehot = train[f'{id}_onehot'][()]
            summaries = train[f'{id}_summaries']
            coord = train[f'{id}_coordinates'][()]
            classes = train[f'{id}_classes'][()]
            img = train[f'{id}_images'][()]
            clusters = cluster_train[str(np.argmax(coord_onehot))]


        elif id_type[0]=='test':
            coord_onehot = test[f'{id}_onehot'][()]
            summaries = test[f'{id}_summaries']
            coord = test[f'{id}_coordinates'][()]
            classes = test[f'{id}_classes'][()]
            img = test[f'{id}_images'][()]
            clusters = cluster_test[str(np.argmax(coord_onehot))]

        else:
            coord_onehot = val[f'{id}_onehot'][()]
            summaries = val[f'{id}_summaries']
            coord = val[f'{id}_coordinates'][()]
            classes = val[f'{id}_classes'][()]
            img = val[f'{id}_images'][()]
            clusters = cluster_val[str(np.argmax(coord_onehot))]


        # save values for MLM_flair
        save_hdf5(all_hdf5[id_type[0]][0], id, img, summaries, coord_onehot, coord, classes, clusters)

        toc = time.perf_counter()
        print(f'====> Finished id {id} -- {((i + 1) / len(id_type[1])) * 100:.2f}% -- {toc - tic:0.2f}s -- {id_type[0]}')

        # save keys
    all_hdf5[id_type[0]][0].create_dataset(name=f'ids', data=np.array(id_type[1], dtype=np.int), compression="gzip", compression_opts=9)

    # close writing hdf5 files
    all_hdf5[id_type[0]][0].close()










