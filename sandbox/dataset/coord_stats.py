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
parser.add_argument('--dataset', default='MLM_v1_sample', type=str,
                        choices=['MLM_v1', 'MLM_v1_sample', 'MLM_v2'], help='dataset')
args = parser.parse_args()

assert args.dataset in ['MLM_v1', 'MLM_v1_sample', 'MLM_v2']

data = []
for i in range(1, 21):
    data_path = f'{str(ROOT_PATH)}/mlm_dataset/{args.dataset}/MLM_{i}/MLM_{i}.json'
    with open(data_path) as json_file:
        data.extend(json.load(json_file))

coords = []
for d in data:
    coord = d['coordinates']
    coord = np.array(coord, dtype=np.float32)
    coords.append((coord[0], coord[1]))

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
print(f'Other: {cont_sorted["NA"] + cont_sorted["AS"] + cont_sorted["AF"] + cont_sorted["SA"] + cont_sorted["OC"] + cont_sorted["AQ"]}')