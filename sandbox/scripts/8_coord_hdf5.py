'''
Create HDF5 file coordinates embeddings
'''
import os
import sys
import json
import h5py
import argparse
import numpy as np
import s2sphere as s2
from time import time
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from collections import Counter

def _init_parallel(img, level):
    cell = s2.Cell.from_lat_lng(s2.LatLng.from_degrees(img[1], img[2]))
    hexid = cell.id().parent(level).to_token()
    return [*img, hexid, cell]


def init_cells(img_container_0, level):
    start = time()
    f = partial(_init_parallel, level=level)
    img_container = []
    with Pool(8) as p:
        for x in p.imap_unordered(f, img_container_0, chunksize=1000):
            img_container.append(x)
    print(f'Time multiprocessing: {time() - start:.2f}s')
    start = time()
    h = dict(Counter(list(list(zip(*img_container))[3])))
    print(f'Time creating h: {time() - start:.2f}s')

    return img_container, h


def delete_cells(img_container, h, t_min):
    del_cells = {k for k, v in h.items() if v <= t_min}
    h = {k: v for k, v in h.items() if v > t_min}
    img_container_f = []
    for img in img_container:
        hexid = img[3]
        if hexid not in del_cells:
            img_container_f.append(img)
    return img_container_f, h


def gen_subcells(img_container_0, h_0, level, t_max):
    img_container = []
    h = {}
    for img in img_container_0:
        hexid_0 = img[3]
        if h_0[hexid_0] > t_max:
            hexid = img[4].id().parent(level).to_token()
            img[3] = hexid
            try:
                h[hexid] = h[hexid] + 1
            except:
                h[hexid] = 1
        else:
            try:
                h[hexid_0] = h[hexid_0] + 1
            except:
                h[hexid_0] = 1
        img_container.append(img)
    return img_container, h

args_lvl_min = 2
args_lvl_max = 10
args_img_min = 20
args_img_max = 500

ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent.parent

# Add arguments to parser
parser = argparse.ArgumentParser(description='Generate MLM entities')
parser.add_argument('--chunk', default=1, type=int, help='number of chunk')
parser.add_argument('--dataset', default='MLM_v1', type=str,
                        choices=['MLM_v1', 'MLM_v1_sample', 'MLM_v2'], help='dataset')
parser.add_argument('--geo_representative', action='store_false')
args = parser.parse_args()

assert args.chunk in range(1, 21)
assert args.dataset in ['MLM_v1', 'MLM_v1_sample', 'MLM_v2']

# read ids if geo-representative
geo_representative_ids = set()
if args.geo_representative:
    with open('eu_ids.json') as json_file:
        geo_representative_ids = set(json.load(json_file))

# load data
img_container = []
for i in range(1, 21):
    data_path = f'{str(ROOT_PATH)}/mlm_dataset/{args.dataset}/MLM_{i}/MLM_{i}.json'
    with open(data_path) as json_file:
        for d in json.load(json_file):
            if args.geo_representative and d['id'] not in geo_representative_ids:
                continue
            img_container.append([d['id'], float(d['coordinates'][0]), float(d['coordinates'][1])])

num_images = len(img_container)
print(f'{num_images} entities available.')
level = args_lvl_min

# initialize
print(f'Initialize cells of level {args_lvl_min} ...')
start = time()
img_container, h = init_cells(img_container, level)
print(f'Time: {time() - start:.2f}s - Number of classes: {len(h)}')

print('Remove cells with |img| < t_min ...')
start = time()
img_container, h = delete_cells(img_container, h, args_img_min)
print(f'Time: {time() - start:.2f}s - Number of classes: {len(h)}')

print('Create subcells ...')
while any(v > args_img_max for v in h.values()) and level < args_lvl_max:
    level = level + 1
    print(f'Level {level}')
    start = time()
    img_container, h = gen_subcells(img_container, h, level, args_img_max)
    print(f'Time: {time() - start:.2f}s - Number of classes: {len(h)}')

print('Remove cells with |img| < t_min ...')
start = time()
img_container, h = delete_cells(img_container, h, args_img_min)
print(f'Time: {time() - start:.2f}s - Number of classes: {len(h)}')
print(f'Number of entities: {len(img_container)}')


# calculate mean GPS coordinate in each cell
coords_sum = {}
for img in img_container:
    if img[3] not in coords_sum:
        coords_sum[img[3]] = [0, 0]
    coords_sum[img[3]][0] = coords_sum[img[3]][0] + img[1]
    coords_sum[img[3]][1] = coords_sum[img[3]][1] + img[2]

# write partitioning information
cell_classes = {}
for i, (k, v) in enumerate(h.items()):
    cell_classes[k] = {
        'class': i,
        'count': v,
        'coordinates': [coords_sum[k][0] / v, coords_sum[k][1] / v]
    }

# write results
if args.geo_representative:
    HDF5_DIR = ROOT_PATH / f'mlm_dataset/hdf5/{args.dataset}/coordinates/coordinates_eu.h5'
else:
    HDF5_DIR = ROOT_PATH / f'mlm_dataset/hdf5/{args.dataset}/coordinates/coordinates.h5'

# hex ids for http://s2.sidewalklabs.com/regioncoverer/
hex_ids = []
with h5py.File(HDF5_DIR, 'w') as h5f:
    for ic in img_container:
        hex_ids.append(ic[3])
        one_hot = np.squeeze(np.eye(len(cell_classes.keys()))[np.array(cell_classes[ic[3]]['class']).reshape(-1)]).astype('float32')
        coordinates = np.array(cell_classes[ic[3]]['coordinates'], dtype='float32')
        # we save 2 values: one hot vector and coordinates
        h5f.create_dataset(name=f'{ic[0]}_onehot', data=one_hot, compression="gzip", compression_opts=9)
        h5f.create_dataset(name=f'{ic[0]}_coordinates', data=coordinates, compression="gzip", compression_opts=9)

    # save keys as int
    h5f.create_dataset(name=f'ids', data=np.array([d[0] for d in img_container], dtype=np.int), compression="gzip", compression_opts=9)

print(','.join(hex_ids))
print('Done!')