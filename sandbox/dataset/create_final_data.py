import os
import sys
import time
import json
import numpy as np
from glob import glob
from pathlib import Path
from collections import Counter

ROOT_PATH = Path(os.path.dirname(__file__))

count = 0
for i in range(1, 21):
    # read images
    image_paths = glob(str(ROOT_PATH / f'MLM_dataset/MLM_{i}/images_{i}/*'))

    image_data = {}
    for im in image_paths:
        # for im in glob(f'{folder}/*'):
            id = im.rsplit('/', 1)[-1].split('_')[0]
            if id in image_data:
                image_data[id].append(im.rsplit('/', 1)[-1])
            else:
                image_data[id] = [im.rsplit('/', 1)[-1]]

    ids = set(image_data.keys())
    assert len(ids) == len(image_data.keys())

    # read coordinates
    coordinate_paths = [ROOT_PATH / f'unique/unique_{i}.json' for i in range(1, 21)]
    coordinate_data = {}
    for p in coordinate_paths:
        with open(p) as json_file:
            coordinate_data.update(json.load(json_file))

    coordinate_data = {k: list(v['coord'][v['coord'].find("(")+1:v['coord'].find(")")].split(' ')[::-1]) for k, v in coordinate_data.items() if k in ids}
    ids = set(coordinate_data.keys())
    assert len(coordinate_data) == len(ids)

    # read summaries
    summaries_paths = [ROOT_PATH / f'summaries/summaries_{i}.json' for i in range(1, 21)]
    summaries_data = {}
    for p in summaries_paths:
        with open(p) as json_file:
            summaries_data.update(json.load(json_file))

    summaries_data = {k: v for k, v in summaries_data.items() if k in ids and len(v['en_wiki']) > 50 and len(v['de_wiki']) > 50 and len(v['fr_wiki']) > 50}
    ids = set(summaries_data.keys())
    assert len(summaries_data) == len(ids)

    # read classes
    triple_paths = [ROOT_PATH / f'triples/triples_{i}.json' for i in range(1, 21)]
    triple_data = {}
    for p in triple_paths:
        with open(p) as json_file:
            triple_data.update(json.load(json_file))

    triple_data = {k: v for k, v in triple_data.items() if k in ids}
    ids = set(triple_data.keys())
    assert len(triple_data) == len(ids)

    label_data = {}
    class_data = {}
    for t in triple_data.keys():
        cl = []
        label_data[t] = triple_data[t][0][0] # get entity english label
        for c in triple_data[t]:
            cl.append(c[2]) # get wikidata class
        class_data[t] = cl

    assert len(ids) == len(label_data)
    assert len(ids) == len(class_data)

    mlm_dataset = []
    for id in ids:
        mlm_dataset.append({
            'id': int(id.strip('Q')),
            'item': label_data[id],
            'coordinates': coordinate_data[id],
            'summaries': summaries_data[id],
            'images': image_data[id],
            'classes': class_data[id]
        })

    # mlm_path = ROOT_PATH / f'MLM_dataset/MLM_{i}/MLM_{i}.json'
    # with open(mlm_path, 'w') as json_file:
    #     json.dump(mlm_dataset, json_file, ensure_ascii=False, indent=4)

    count += len(mlm_dataset)
    print(f'Finished chunk {i}')
    print(len(mlm_dataset))
    print(count)
