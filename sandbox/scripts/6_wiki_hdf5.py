'''
Create HDF5 file with wiki text embeddings
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
# parser.add_argument('--chunk', default=1, type=int, help='number of chunk')
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
HDF5_DIR = ROOT_PATH / f'mlm_dataset/hdf5/summaries/summaries_{args.embedding}.h5'
print(f'Getting {args.embedding} embeddings and writing data into HDF5...')
tic = time.perf_counter()
with h5py.File(HDF5_DIR, 'a') as h5f:
    for i, d in enumerate(data):
        id = d['id']
        print(f'Getting results for id {id}')
        # Get values
        try:
            if f'{id}_en' not in h5f:
                en_embedding = doc_embeddings.embed(d['summaries']['en_wiki'])
                h5f.create_dataset(name=f'{id}_en', data=en_embedding.detach().cpu().numpy(), compression="gzip", compression_opts=9)
            if f'{id}_de' not in h5f:
                de_embedding = doc_embeddings.embed(d['summaries']['fr_wiki'])
                h5f.create_dataset(name=f'{id}_de', data=de_embedding.detach().cpu().numpy(), compression="gzip", compression_opts=9)
            if f'{id}_fr' not in h5f:
                fr_embedding = doc_embeddings.embed(d['summaries']['de_wiki'])
                h5f.create_dataset(name=f'{id}_fr', data=fr_embedding.detach().cpu().numpy(), compression="gzip", compression_opts=9)
            toc = time.perf_counter()
            print(f'====> Finished id {id} -- {((i+1)/len(data))*100:.2f}% -- {toc - tic:0.2f}s')
        except Exception as e:
            print(f'Skipping id {id}')
            print(str(e))
    # save keys as int
    h5f.create_dataset(name=f'ids', data=np.array([d['id'] for d in data], dtype=np.int), compression="gzip", compression_opts=9)
print('Done!')
