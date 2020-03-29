'''
Create HDF5 file with classes embeddings
'''
import os
import time
import h5py
import json
import argparse
import numpy as np
from pathlib import Path
from doc_embeddings import DocEmbeddings

ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent.parent

# Add arguments to parser
parser = argparse.ArgumentParser(description='Generate MLM entities')
parser.add_argument('--embedding', default='bert', type=str,
                        choices=['flair', 'bert'], help='model to train the dataset')
args = parser.parse_args()

# assert args.chunk in range(1, 21)
assert args.embedding in ['flair', 'bert']

# load data
data = []
for i in range(1, 21):
    data_path = f'{str(ROOT_PATH)}/mlm_dataset/MLM_dataset/MLM_{i}/MLM_{i}.json'
    with open(data_path) as json_file:
        data.extend(json.load(json_file))

doc_embeddings = DocEmbeddings(args.embedding)

# write data
HDF5_DIR = ROOT_PATH / f'mlm_dataset/hdf5/classes/classes_{args.embedding}.h5'
print(f'Getting {args.embedding} embeddings and writing data into HDF5...')
tic = time.perf_counter()
with h5py.File(HDF5_DIR, 'w') as h5f:
    for i, d in enumerate(data):
        id = d['id']
        print(f'Getting results for id {id}')
        # Get values
        all_classes = []
        for cl in d['classes']:
            embeddings = doc_embeddings.embed(cl)
            all_classes.append(embeddings.detach().cpu().numpy())

        h5f.create_dataset(name=f'{id}_classes', data=np.stack([all_classes], axis=0), compression="gzip", compression_opts=9)
        toc = time.perf_counter()
        print(f'====> Finished id {id} -- {((i+1)/len(data))*100:.2f}% -- {toc - tic:0.2f}s')
    # save keys as int
    h5f.create_dataset(name=f'ids', data=np.array([d['id'] for d in data], dtype=np.int), compression="gzip", compression_opts=9)
print('Done!')
