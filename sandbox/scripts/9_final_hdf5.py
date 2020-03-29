'''
Create one HDF5 file and store whole dataset
'''
import os
import h5py
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# define paths
ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent
HDF5_DIR = ROOT_PATH.parent / 'mlm_dataset/hdf5'

# read data
images = h5py.File(os.path.join(HDF5_DIR, 'images/images.h5'), 'r')
images_cropped = h5py.File(os.path.join(HDF5_DIR, 'images/images_cropped.h5'), 'r')
summaries_flair = h5py.File(os.path.join(HDF5_DIR, 'summaries/summaries_flair.h5'), 'r')
summaries_bert = h5py.File(os.path.join(HDF5_DIR, 'summaries/summaries_bert.h5'), 'r')
coordinates = h5py.File(os.path.join(HDF5_DIR, 'coordinates/coordinates.h5'), 'r')
classes = h5py.File(os.path.join(HDF5_DIR, 'classes/classes_bert.h5'), 'r')

ids = list(coordinates['ids'])

# split to train, test and val
# 80% training, 10% validation and 10% test
train_ids, val_ids = train_test_split(ids, test_size=0.2, shuffle=True)
val_ids, test_ids = train_test_split(val_ids, test_size=0.5, shuffle=False)

# MLM flair
MLM_flair_train = h5py.File(os.path.join(ROOT_PATH, 'dataset/flair/train.h5'), 'w')
MLM_flair_val = h5py.File(os.path.join(ROOT_PATH, 'dataset/flair/val.h5'), 'w')
MLM_flair_test = h5py.File(os.path.join(ROOT_PATH, 'dataset/flair/test.h5'), 'w')
# ---------
# MLM flair cropped
MLM_flair_cropped_train = h5py.File(os.path.join(ROOT_PATH, 'dataset/flair_cropped/train.h5'), 'w')
MLM_flair_cropped_val = h5py.File(os.path.join(ROOT_PATH, 'dataset/flair_cropped/val.h5'), 'w')
MLM_flair_cropped_test = h5py.File(os.path.join(ROOT_PATH, 'dataset/flair_cropped/test.h5'), 'w')
# ---------
# MLM BERT
MLM_bert_train = h5py.File(os.path.join(ROOT_PATH, 'dataset/bert/train.h5'), 'w')
MLM_bert_val = h5py.File(os.path.join(ROOT_PATH, 'dataset/bert/val.h5'), 'w')
MLM_bert_test = h5py.File(os.path.join(ROOT_PATH, 'dataset/bert/test.h5'), 'w')
# ---------
# MLM BERT cropped
MLM_bert_cropped_train = h5py.File(os.path.join(ROOT_PATH, 'dataset/bert_cropped/train.h5'), 'w')
MLM_bert_cropped_val = h5py.File(os.path.join(ROOT_PATH, 'dataset/bert_cropped/val.h5'), 'w')
MLM_bert_cropped_test = h5py.File(os.path.join(ROOT_PATH, 'dataset/bert_cropped/test.h5'), 'w')
# ---------

def save_hdf5(h5f, id, im, sum, coord, one_hot, cl):
    h5f.create_dataset(name=f'{id}_images', data=im, compression="gzip", compression_opts=9)
    h5f.create_dataset(name=f'{id}_summaries', data=sum, compression="gzip", compression_opts=9)
    h5f.create_dataset(name=f'{id}_coordinates', data=coord, compression="gzip", compression_opts=9)
    h5f.create_dataset(name=f'{id}_onehot', data=one_hot, compression="gzip", compression_opts=9)
    h5f.create_dataset(name=f'{id}_classes', data=cl, compression="gzip", compression_opts=9)

all_hdf5 = {
    'train': [MLM_flair_train, MLM_flair_cropped_train, MLM_bert_train, MLM_bert_cropped_train],
    'val': [MLM_flair_val, MLM_flair_cropped_val, MLM_bert_val, MLM_bert_cropped_val],
    'test': [MLM_flair_test, MLM_flair_cropped_test, MLM_bert_test, MLM_bert_cropped_test]
}

all_ids = [('train', train_ids), ('val', val_ids), ('test', test_ids)]

tic = time.perf_counter()
for id_type in all_ids:
    for i, id in enumerate(id_type[1]):
        img = images[f'{id}_images'][()]
        img_cropped = images_cropped[f'{id}_images'][()]
        sum_flair_en = summaries_flair[f'{id}_en'][()]
        sum_flair_de = summaries_flair[f'{id}_de'][()]
        sum_flair_fr = summaries_flair[f'{id}_fr'][()]
        sum_bert_en = summaries_bert[f'{id}_en'][()]
        sum_bert_de = summaries_bert[f'{id}_de'][()]
        sum_bert_fr = summaries_bert[f'{id}_fr'][()]
        coord = coordinates[f'{id}_coordinates'][()]
        coord_onehot = coordinates[f'{id}_onehot'][()]
        cl = classes[f'{id}_classes'][()]

        # save values for MLM_flair
        save_hdf5(all_hdf5[id_type[0]][0], id, img, np.stack([sum_flair_en, sum_flair_de, sum_flair_fr], axis=0), coord, coord_onehot, np.squeeze(cl, axis=0))
        # save values for MLM_flair_cropped
        save_hdf5(all_hdf5[id_type[0]][1], id, img_cropped, np.stack([sum_flair_en, sum_flair_de, sum_flair_fr], axis=0), coord, coord_onehot, np.squeeze(cl, axis=0))
        # save values for MLM_bert
        save_hdf5(all_hdf5[id_type[0]][2], id, img, np.stack([sum_bert_en, sum_bert_de, sum_bert_fr], axis=0), coord, coord_onehot, np.squeeze(cl, axis=0))
        # save values for MLM_bert_cropped
        save_hdf5(all_hdf5[id_type[0]][3], id, img_cropped, np.stack([sum_bert_en, sum_bert_de, sum_bert_fr], axis=0), coord, coord_onehot, np.squeeze(cl, axis=0))

        toc = time.perf_counter()
        print(f'====> Finished id {id} -- {((i+1)/len(id_type[1]))*100:.2f}% -- {toc - tic:0.2f}s -- {id_type[0]}')

    # save keys
    all_hdf5[id_type[0]][0].create_dataset(name=f'ids', data=np.array(id_type[1], dtype=np.int), compression="gzip", compression_opts=9)
    all_hdf5[id_type[0]][1].create_dataset(name=f'ids', data=np.array(id_type[1], dtype=np.int), compression="gzip", compression_opts=9)
    all_hdf5[id_type[0]][2].create_dataset(name=f'ids', data=np.array(id_type[1], dtype=np.int), compression="gzip", compression_opts=9)
    all_hdf5[id_type[0]][3].create_dataset(name=f'ids', data=np.array(id_type[1], dtype=np.int), compression="gzip", compression_opts=9)

    # close writing hdf5 files
    all_hdf5[id_type[0]][0].close()
    all_hdf5[id_type[0]][1].close()
    all_hdf5[id_type[0]][2].close()
    all_hdf5[id_type[0]][3].close()

# close reading hdf5 files
images.close()
images_cropped.close()
summaries_flair.close()
summaries_bert.close()
coordinates.close()
classes.close()