import os
import sys
import time
import json
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent.parent


triple_paths = [ROOT_PATH / f'mlm_dataset/MLM_dataset/MLM_{i}/MLM_{i}.json' for i in range(1, 21)]

triple_classes = []
data = []

for p in triple_paths:
    with open(p) as json_file:
        f = json.load(json_file)
        data.extend(d for d in f)
        triple_classes.extend([c for d in f for c in d['classes']])


unique_classes = list(set(triple_classes))

classes_counter = {c:0 for c in unique_classes}

for tc in triple_classes:
    classes_counter[tc] += 1

sorted_classes = {k: v for k, v in sorted(classes_counter.items(), key=lambda item: item[1], reverse=True)}

# [print(f'{k}: {v}') for k, v in sorted_classes.items()]

num = 1000
# top = set(list(sorted_classes.keys())[:num])

[print(f'{k}: {v}') for i, (k, v) in enumerate(sorted_classes.items()) if i < num]
print(len(data))