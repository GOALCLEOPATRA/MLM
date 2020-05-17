"""
https://en.wikipedia.org/wiki/List_of_European_Union_member_states_by_population

Code Country
------------
BE Belgium
BG Bulgaria
CZ Czechia
DK Denmark
DE Germany
EE Estonia
IE Ireland
GR Greece
ES Spain
FR France
HR Croatia
IT Italy
CY Cyprus
LV Latvia
LT Lithuania
LU Luxembourg
HU Hungary
MT Malta
NL Netherlands
AT Austria
PL Poland
PT Portugal
RO Romania
SI Slovenia
SK Slovakia
FI Finland
SE Sweden
GB United Kingdom
"""
import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent.parent

# Add arguments to parser
parser = argparse.ArgumentParser(description='Generate MLM entities')
parser.add_argument('--dataset', default='MLM_v1', type=str,
                        choices=['MLM_v1', 'MLM_v1_sample', 'MLM_v2'], help='dataset')
args = parser.parse_args()

assert args.dataset in ['MLM_v1', 'MLM_v1_sample', 'MLM_v2']

data = []
for i in range(1, 21):
    data_path = f'{str(ROOT_PATH)}/mlm_dataset/{args.dataset}/MLM_{i}/MLM_{i}.json'
    with open(data_path) as json_file:
        data.extend(json.load(json_file))

import reverse_geocoder
from collections import Counter
import random
random.seed(1234)

coords = []
ids = []
for d in data:
    coord = d['coordinates']
    coord = np.array(coord, dtype=np.float32)
    coords.append((coord[0], coord[1]))
    ids.append(d['id'])

isos = reverse_geocoder.search(coords)

ids_iso = list(zip(ids, [iso['cc'] for iso in isos]))

eu_ids_per = {
    'BE': 2.23,
    'BG': 1.36,
    'CZ': 2.07,
    'DK': 1.13,
    'DE': 16.17,
    'EE': 0.26,
    'IE': 0.96,
    'GR': 2.09,
    'ES': 9.14,
    'FR': 13.05,
    'HR': 0.79,
    'IT': 11.76,
    'CY': 0.17,
    'LV': 0.37,
    'LT': 0.54,
    'LU': 0.12,
    'HU': 1.90,
    'MT': 0.10,
    'NL': 3.37,
    'AT': 1.73,
    'PL': 7.40,
    'PT': 2.00,
    'RO': 3.78,
    'SI': 0.41,
    'SK': 1.06,
    'FI': 1.07,
    'SE': 1.99,
    'GB': 12.98
}

"""
We have least number of entities for Malta (24)
Then our total number of entities will be: (24x100)/0.1 = 24000
For finding number of entities for others we do (perx24000)/100
"""
total_entities = 24000

import pycountry_convert as pc
eu_iso = []
eu_ids = []
eu_ids_iso = {}
for id, iso in ids_iso:
    if iso in set(eu_ids_per.keys()):
        if iso not in eu_ids_iso:
            eu_ids_iso[iso] = []
        eu_ids_iso[iso].append(id)

final_ids = {}
final_iso = []
for iso, ids in eu_ids_iso.items():
    num_allowed_ids = int((eu_ids_per[iso]*total_entities)/100)
    if num_allowed_ids > len(ids):
        print(f'{iso} -> required: {num_allowed_ids}, got: {len(ids)}')
        final_ids[iso] = ids
    else:
        final_ids[iso] = random.sample(ids, num_allowed_ids)
    final_iso.extend([iso for _ in final_ids[iso]])

cont_dict = Counter(final_iso)

cont_sorted = {k: v for k, v in sorted(cont_dict.items(), key=lambda item: item[1], reverse=True)}

[print(f'{k}: {v}') for k, v in cont_sorted.items()]
print(f'EU countries: {len(cont_sorted.keys())}')
print(f'Total HS: {len(final_iso)}')

ids_to_save = [id for l in final_ids.values() for id in l]

with open('eu_ids.json', 'w') as outfile:
    json.dump(ids_to_save, outfile, ensure_ascii=False, indent=4)