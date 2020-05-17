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
ids_iso = []
for d in data:
    coord = d['coordinates']
    coord = np.array(coord, dtype=np.float32)
    coords.append((coord[0], coord[1]))
    ids_iso.append(d['id'])

isos = reverse_geocoder.search(coords)

ids_iso = list(zip(ids_iso, [iso['cc'] for iso in isos]))

"""
FR: 38793 - 40%
US: 34463 - 100%
DE: 31654 - 40%
PL: 19433 - 30%
GB: 13042 - 60%
IT: 9355 - 60%
ES: 8408 - 60%
CZ: 7742 - 25%
HU: 3048 - 60%
NL: 2954 - 30%
SI: 2813 - 40%
AT: 2639 - 60%
SK: 1985 - 50%
BE: 1603 - 70%
"""
geo_representative_ids = []
iso = []
for id, i in ids_iso:
    if i == 'FR' and random.uniform(0, 1) > 0.10: # FR: 38793 - 10%
        continue
    if i == 'DE' and random.uniform(0, 1) > 0.10: # DE: 31654 - 10%
        continue
    if i == 'PL' and random.uniform(0, 1) > 0.1: # PL: 19433 - 10%
        continue
    if i == 'GB' and random.uniform(0, 1) > 0.2: # GB: 13042 - 20%
        continue
    if i == 'IT' and random.uniform(0, 1) > 0.2: # IT: 9355 - 20%
        continue
    if i == 'ES' and random.uniform(0, 1) > 0.2: # ES: 8408 - 20%
        continue
    if i == 'CZ' and random.uniform(0, 1) > 0.1: # CZ: 7742 - 10%
        continue
    if i == 'HU' and random.uniform(0, 1) > 0.3: # HU: 3048 - 30%
        continue
    if i == 'NL' and random.uniform(0, 1) > 0.3: # NL: 2954 - 30%
        continue
    if i == 'SI' and random.uniform(0, 1) > 0.3: # SI: 2813 - 30%
        continue
    if i == 'AT' and random.uniform(0, 1) > 0.3: # AT: 2639 - 30%
        continue
    if i == 'SK' and random.uniform(0, 1) > 0.3: # SK: 1985 - 30%
        continue
    if i == 'BE' and random.uniform(0, 1) > 0.3: # BE: 1603 - 30%
        continue

    geo_representative_ids.append(id)
    iso.append(i)

import pycountry_convert as pc
continents = []
final_ids = []
for i, id in zip(iso, geo_representative_ids):
    try:
        cont = pc.country_alpha2_to_continent_code(i)
        if cont == 'EU' and random.uniform(0, 1) > 0.3:
            continue
        if cont == 'NA' and random.uniform(0, 1) > 0.5:
            continue
        continents.append(cont)
        final_ids.append(id)
    except:
        if i == 'VA':
            continents.append('EU')
        elif i == 'PN':
            continents.append('OC')
        elif i == 'SX':
            continents.append('NA')
        elif i == 'EH':
            continents.append('AF')
        elif i == 'AQ':
            continents.append('AQ')
        elif i == 'TF':
            continents.append('AQ')
        elif i == 'TL':
            continents.append('AS')
        else:
            print(f'No alpha code 2 for {i}')

print(len(final_ids))
cont_dict = Counter(continents)

cont_sorted = {k: v for k, v in sorted(cont_dict.items(), key=lambda item: item[1], reverse=True)}

[print(f'{k}: {v}') for k, v in cont_sorted.items()]
print(len(cont_sorted.keys()))

print(f'EU: {cont_sorted["EU"]}')
print(f'Other: {cont_sorted["NA"] + cont_sorted["AS"] + cont_sorted["AF"] + cont_sorted["SA"] + cont_sorted["OC"] + cont_sorted["AQ"]}')

with open('geo_representative_ids.json', 'w') as outfile:
    json.dump(final_ids, outfile, ensure_ascii=False, indent=4)