'''
Create one HDF5 file and store triples
'''
import os
import h5py
import json
import torch
import random
import numpy as np
from pathlib import Path
import datetime as dt

tik = dt.datetime.now()

# Define paths
ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent.parent
HDF5_DIR = ROOT_PATH / 'dataset'
SOURCE = ROOT_PATH / 'dataset_files/pilot/'
wikidata_path = os.path.join(SOURCE, 'capitals_wikidata.json')

# Read data from json
wikidata = []
with open(wikidata_path) as json_file:
    wikidata = json.load(json_file)


# Create triple dictionary
print('Processing with triples...')
# We take only english for now - We do triples later
triple_dict = {}
for sample in wikidata:
    key = sample['item'].rsplit('/', 1)[-1]
    triple_dict[key] = sample['wikidata']['en']
print('Done!')


# write train data
print('Writing data into HDF5 files...')
for key_label in triple_dict.keys():
    with h5py.File(os.path.join(HDF5_DIR, f'{key_label}_pilot.h5'), 'w') as h5f:
        for i, key in enumerate(triple_dict[key_label]):
            # save values
            h5f.create_dataset(name=f'{key.strip("Q")}_triples', data=triple_dict[key], compression="gzip", compression_opts=9)

        # save keys as int
        h5f.create_dataset(name=f'ids', data=np.array([key.strip('Q') for key in triple_dict[key_label]], dtype=np.int), compression="gzip", compression_opts=9)
print('Done!')

tok = dt.datetime.now()

print(f'Total time: {tok-tik} seconds')
