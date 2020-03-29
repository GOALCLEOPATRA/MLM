import os
import json
import h5py
import time
import numpy as np
from glob import glob
from pathlib import Path

# define paths
ROOT_PATH = Path(os.path.dirname(__file__))
coordinates = h5py.File(ROOT_PATH / 'hdf5/coordinates/coordinates.h5', 'r')

ids = set(coordinates['ids'])

print(len(ids))
count = 0
for i in range(1, 21):
    image_path = glob(str(ROOT_PATH / f'MLM_dataset/MLM_{i}/images_{i}/*'))
    image_path_cropped = glob(str(ROOT_PATH / f'MLM_dataset(cropped_images)/MLM_{i}/images_{i}/*'))
    json_path_cropped = ROOT_PATH / f'MLM_dataset(cropped_images)/MLM_{i}/MLM_{i}.json'
    json_data_cropped = []

    with open(json_path_cropped) as json_file:
        json_data_cropped = json.load(json_file)

    allowed_images = set([img for d in json_data_cropped for img in d['images']])

    not_allowed = []
    for j, img in enumerate(image_path):
        if img.rsplit('/', 1)[-1] not in allowed_images:
            not_allowed.append(img)

    for j, img in enumerate(image_path_cropped):
        if img.rsplit('/', 1)[-1] not in allowed_images:
            not_allowed.append(img)

    count += len(allowed_images)
    # delete non allowed images
    # for f in not_allowed:
    #     os.remove(f)

print(f'Total images: {count}')