'''
Create one HDF5 file and store coordinates
'''
import os
import h5py
import json
import numpy as np
from pathlib import Path
import datetime as dt

tik = dt.datetime.now()

# Define paths - fix path locally
ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent.parent
HDF5_DIR = ROOT_PATH / 'dataset'
SOURCE = ROOT_PATH / 'dataset_files/pilot/'
data_path = os.path.join(SOURCE, 'capitals_final.json')

# Read data from json
data = []
with open(data_path) as json_file:
    data = json.load(json_file)

# # Create data dictionary
print('Processing with text and coordinates...')
coord_dict = {}
for sample in data:
    key = sample['item'].rsplit('/', 1)[-1]
    # Get values
    coord = sample['coord'][sample['coord'].find("(")+1:sample['coord'].find(")")].split(' ')

    # Process values
    coord = np.array(coord, dtype=np.float32)

    # Save values
    coord_dict[key] = coord
print('Done!')

# write train data
print('Writing data into HDF5 files...')
for key_label in coord_dict.keys():
    with h5py.File(os.path.join(HDF5_DIR, f'{key_label}_pilot.h5'), 'w') as h5f:
        for i, key in enumerate(coord_dict[key_label]):
            # save values
            h5f.create_dataset(name=f'{key.strip("Q")}_coords', data=coord_dict[key], compression="gzip", compression_opts=9)

        # save keys as int
        h5f.create_dataset(name=f'ids', data=np.array([key.strip('Q') for key in coord_dict[key_label]], dtype=np.int), compression="gzip", compression_opts=9)
print('Done!')

tok = dt.datetime.now()

print(f'Total time: {tok-tik} seconds')
