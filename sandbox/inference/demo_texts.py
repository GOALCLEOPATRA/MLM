from __future__ import print_function
import os
import sys
import json
import h5py
import torch
import numpy as np
from pathlib import Path
from more_itertools import unique_everseen

# set paths
ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent
h5f = h5py.File(ROOT_PATH / 'dataset/bert/train.h5', 'r')

# set inference ids
inf_ids = [23178, 630749, 10958, 23048167, 172, 1789630,
           1061703, 328000, 1062, 20962888, 680361, 112019,
           1331632, 736060, 1744197, 55954790, 534742, 691876, 549700, 955]

ids = set(h5f['ids'][()])
assert len(ids) == len(list(ids))

# check difference and intersection with ttraining set
difference = set(inf_ids).difference(ids)
intersection = set(inf_ids).intersection(ids)
print(f'Diffence: {difference}')
print(f'Intersection: {intersection}')

# load model for image
sys.path.append(str(ROOT_PATH))
from models.model import MLMRetrieval
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLMRetrieval()
model.to(device)
checkpoint = torch.load(f'{ROOT_PATH}/experiments/snapshots/LSTM_image_model_e8_v-137.60.pth.tar', encoding='latin1')
model.load_state_dict(checkpoint['state_dict'])

# pre calculate one hot vectors
onehot_vec = torch.stack([torch.from_numpy(np.squeeze(np.eye(436)[np.array(i).reshape(-1)]).astype('float32')).to(device) for i in range(436)])
coord_embed = model.learn_coord(onehot_vec)
coord_embed = [ce.data.cpu().numpy() for ce in torch.unbind(coord_embed)]

cordinates = {}
for id in ids:
    index = np.argmax(h5f[f'{id}_onehot'][()])
    if index not in cordinates:
        cordinates[index] = h5f[f'{id}_coordinates'][()].tolist()

    # if len(cordinates.keys()) == 436:
    #     break

# load data
coord_classes = np.array([i for i in range(436)])
text = {}
coords = {}
onehot = {}
for id in intersection:
    text[id] = h5f[f'{id}_summaries'][()]
    coords[id] = h5f[f'{id}_coordinates'][()]
    onehot[id] = h5f[f'{id}_onehot'][()]

# get text embeddings and calculate similarities with coord embeddings
results = {}
for k in text.keys():
    txt = text[k]
    coord_cls = np.argmax(onehot[k])
    data = {}
    for i, tx in enumerate(txt):
        tx = model.learn_sum(torch.from_numpy(tx).unsqueeze(0).to(device)).squeeze(0).data.cpu().numpy()
        similarities = [np.dot(tx, cm) for cm in coord_embed]

        sorting = np.argsort(similarities)[::-1].tolist()
        sorting = coord_classes[sorting].tolist()
        # sorting = list(unique_everseen(sorting))

        pos = sorting.index(coord_cls)

        if i == 0: key = 'en'
        if i == 1: key = 'de'
        if i == 2: key = 'fr'

        data[key] = {
            'true_coordinate': cordinates[coord_cls],
            'model_prediction': cordinates[sorting[0]],
            'true_coordinate_rank': pos,
            'top_5': [cordinates[s] for s in sorting[:5]]
        }
    results[f'Q{k}_{i}'] = data

with open('texts_demo.json', 'w') as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)