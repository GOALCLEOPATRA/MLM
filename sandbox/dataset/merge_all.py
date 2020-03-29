import os
import sys
import time
import json
import numpy as np
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(__file__)).parent #.parent.parent


triple_paths = [ROOT_PATH / f'mlm_dataset/triples/triples_{i}.json' for i in range(1, 21)]
images_paths = [ROOT_PATH / f'mlm_dataset/images/images_{i}.json' for i in range(1, 21)]
unique_paths = [ROOT_PATH / f'mlm_dataset/unique/unique_{i}.json' for i in range(1, 21)]
triples_keys = []
images_keys = []
unique_data = {}
for i, path in enumerate([triple_paths, images_paths, unique_paths]):
    for p in path:
        with open(p) as json_file:
            if i == 0: triples_keys.extend(list(json.load(json_file).keys()))
            if i == 1: images_keys.extend(list(json.load(json_file).keys()))
            if i == 2: unique_data.update(json.load(json_file))

assert set(triples_keys) == set(images_keys)

print(len(images_keys))
print(len(unique_data.keys()))

coords = []
for k in images_keys:
    coord = unique_data[k]['coord'][unique_data[k]['coord'].find("(")+1:unique_data[k]['coord'].find(")")].split(' ')
    coord = np.array(coord, dtype=np.float32)
    coords.append((coord[1], coord[0]))

print(len(coords))

import reverse_geocoder
from collections import Counter
countries = reverse_geocoder.search(coords)

iso = [c['cc'] for c in countries]

iso_dict = Counter(iso)

iso_sorted = {k: v for k, v in sorted(iso_dict.items(), key=lambda item: item[1], reverse=True)}

[print(f'{k}: {v}') for k, v in iso_sorted.items()]
# print(len(iso_sorted.keys()))

import pycountry_convert as pc

# iso = [i.replace('VA', 'VAT') for i in iso if i == 'VA']
continents = []
for i in iso:
    try:
        if i != 'FR':
            cont = pc.country_alpha2_to_continent_code(i)
            continents.append(cont)
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

cont_dict = Counter(continents)

cont_sorted = {k: v for k, v in sorted(cont_dict.items(), key=lambda item: item[1], reverse=True)}

[print(f'{k}: {v}') for k, v in cont_sorted.items()]
print(len(cont_sorted.keys()))

print(f'EU: {cont_sorted["EU"]}')
print(f'Other: {cont_sorted["NA"] + cont_sorted["AS"] + cont_sorted["AF"] + cont_sorted["SA"] + cont_sorted["OC"]}')