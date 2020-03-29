import os
import sys
import time
import json
from pathlib import Path
from itertools import islice
from SPARQLWrapper import SPARQLWrapper, JSON

ROOT_PATH = Path(os.path.dirname(__file__)).parent #.parent.parent

paths = [ROOT_PATH / f'mlm_dataset/entities/entities_{i}.json' for i in range(1, 21)]
data = []
keys = []
for p in paths:
    with open(p) as json_file:
        data.extend(json.load(json_file))
        keys.extend([d["item"].rsplit('/', 1)[-1] for d in data])

keys = set(keys)
keys_len = len(keys)
unique_data = {}
# write unique entites
for d in data:
    k = d["item"].rsplit('/', 1)[-1]
    if k in keys:
        unique_data[k] = {
            'coord': d['coord'],
            'enlink': d['enlink'],
            'delink': d['delink'],
            'frlink': d['frlink']
        }
        keys.remove(k)

assert keys_len == len(unique_data.keys())

def chunk_list(data, num=20):
    keys_inc = len(data.keys())
    while keys_inc % num != 0:
        keys_inc += 1
    obj_in_dict = int(keys_inc/num)
    it = iter(data)
    for i in range(0, num):
        yield {k:data[k] for k in islice(it, obj_in_dict)}

chunked_data = list(chunk_list(unique_data))

assert keys_len == len(set([k for cd in chunked_data for k in list(cd.keys())]))

mlm_entites_path = ROOT_PATH / 'mlm_dataset/unique'

for i, chunk_d in enumerate(chunked_data):
    chunk_path = f'{mlm_entites_path}/unique_{i+1}.json'
    with open(chunk_path, 'w') as json_file:
        json.dump(chunk_d, json_file, ensure_ascii=False, indent=4)

    print(f'Finished chunk {i+1}')

print(f'Total MLM entities: {len(unique_data.keys())}')
