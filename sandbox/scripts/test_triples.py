import os
import sys
import time
import json
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON

ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent.parent

data_path = ROOT_PATH / 'dataset_files/pilot/capitals_triples.json'
data = []

with open(data_path) as json_file:
    data = json.load(json_file)

first_key = list(data.keys())[0]
all_poperties = set([tpl[4] for tpl in data[first_key] if tpl[2] == 'instance of'])
obj_counter = {}

for k in data.keys():
    # all_poperties.intersection(set([tpl[2] for tpl in data[k] if tpl[2] == 'instance of']))
    obj = [tpl[4] for tpl in data[k] if tpl[2] == 'instance of']
    for o in obj:
        if o not in obj_counter:
            obj_counter[o] = 1
        else:
            obj_counter[o] += 1

# all_poperties = list(all_poperties)
# all_poperties.sort()
# [print(prop) for prop in all_poperties]
all = obj_counter.items()
all = sorted(all, key=lambda x: x[1], reverse=True)
[print(f'{k}: {v}') for k, v in all]