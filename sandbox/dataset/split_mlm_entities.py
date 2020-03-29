import os
import sys
import time
import json
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON

ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent.parent

cities_path = ROOT_PATH / 'mlm_dataset/initial_query.json'
data = []

with open(cities_path) as json_file:
    data = json.load(json_file)

new_data = []
seen_ids = []
for d in data:
	if d['item'] not in seen_ids:
		new_data.append(d)
		seen_ids.append(d['item'])

unique_ids = list(set([d['item'].rsplit('/', 1)[-1] for d in data]))

assert len(seen_ids) == len(unique_ids)

def chunk_list(seq, num=20):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

chunked_cities = chunk_list(new_data)

mlm_entites_path = ROOT_PATH / 'mlm_dataset/initial'

for i, chunk_data in enumerate(chunked_cities):
    chunk_path = f'{mlm_entites_path}/initial_{i+1}.json'
    with open(chunk_path, 'w') as json_file:
        json.dump(chunk_data, json_file, ensure_ascii=False, indent=4)