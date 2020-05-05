from pathlib import Path
import h5py
import os
import numpy as np
import json

ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent
train = h5py.File(f'{ROOT_PATH}/data/train.h5', 'r')
test = h5py.File(f'{ROOT_PATH}/data/test.h5', 'r')
val = h5py.File(f'{ROOT_PATH}/data/val.h5', 'r')
all_ids = [('train', train['ids']), ('val', val['ids']), ('test', test['ids'])]

def save_file(fileName, file):
    with open(fileName, 'w') as outfile:
        json.dump(file, outfile)


def open_json(fileName):
    try:
        with open(fileName,encoding='utf8') as json_data:
            d = json.load(json_data)
    except Exception as s:
        d=s
        print(d)
    return d


def find_n_cells():
    max_cls = 0
    for id_type in all_ids:

        for i, id in enumerate(id_type[1]):
            if id_type[0] == 'train':
                onehot = train[f'{id}_onehot'][()]
            elif id_type[0] == 'test':
                onehot = test[f'{id}_onehot'][()]
            else:
                onehot = val[f'{id}_onehot'][()]
            cls = np.argmax(onehot)
            if cls > max_cls:
                max_cls = cls
    return max_cls + 1


def cluster_coord_cells(n_cells, file, _ids,part):
    print('********** '+part)
    cells = range(0, n_cells)
    clusters = {}

    for cell in cells:

        are_this_cell = []

        for id in _ids:
            onehot = file[f'{id}_onehot'][()]
            cls = np.argmax(onehot)
            if cls == cell:
                are_this_cell.append(int(id))

        clusters[cell] = are_this_cell
        print(cell, len(are_this_cell))

    save_file('clusters_'+str(part), clusters)


# n_cells = find_n_cells()
# print(n_cells)
n_cells = 116
cluster_coord_cells(n_cells, train, train['ids'],'train' )
cluster_coord_cells(n_cells, test, test['ids'],'test' )
cluster_coord_cells(n_cells, val, val['ids'],'val' )
print('done!')