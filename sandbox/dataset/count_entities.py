import os
import sys
import time
import json
from pathlib import Path
from itertools import islice
from SPARQLWrapper import SPARQLWrapper, JSON

ROOT_PATH = Path(os.path.dirname(__file__)).parent #.parent.parent


triple_paths = [ROOT_PATH / f'mlm_dataset/triples/triples_{i}.json' for i in range(1, 21)]
images_paths = [ROOT_PATH / f'mlm_dataset/images/images_{i}.json' for i in range(1, 21)]
data = []
triples_keys = []
images_keys = []
for i, path in enumerate([triple_paths, images_paths]):
    for p in path:
        with open(p) as json_file:
            if i == 0: triples_keys.extend(list(json.load(json_file).keys()))
            if i == 1: images_keys.extend(list(json.load(json_file).keys()))

assert set(triples_keys) == set(images_keys)

print(len(images_keys))