'''
Create one HDF5 file and store wiki text embeddings
'''
import os
import h5py
import json
import torch
import flair
import numpy as np
from pathlib import Path
import datetime as dt
from flair.embeddings import FlairEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence

# set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
flair.device = DEVICE

tik = dt.datetime.now()

# Define paths - fix path locally
ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent.parent
HDF5_DIR = ROOT_PATH / 'dataset'
SOURCE = ROOT_PATH / 'dataset_files/pilot/'
data_path = os.path.join(SOURCE, 'capitals_final.json')

# Read data from json
data = []
with open(data_path) as json_file:
    data = json.load(json_file)

# Initialize BERT embeddings using Flair framework
# bert_embeddings = BertEmbeddings('bert-base-uncased')
print('Loading text embeddings...')
flair_embedding_forward = FlairEmbeddings('news-forward')
# lair_embedding_backward = FlairEmbeddings('news-backward')
document_embeddings = DocumentPoolEmbeddings([flair_embedding_forward])
print('Done!')

# # Create data dictionary
print('Processing with text and coordinates...')
enwiki_dict = {}
frwiki_dict = {}
dewiki_dict = {}
for sample in data:
    key = sample['item'].rsplit('/', 1)[-1]
    # Get values
    en = Sentence(sample['en_wiki']['text'])
    fr = Sentence(sample['fr_wiki']['text'])
    de = Sentence(sample['de_wiki']['text'])

    # Process values
    document_embeddings.embed(en)
    document_embeddings.embed(fr)
    document_embeddings.embed(de)

    # Save values
    enwiki_dict[key] = en.embedding.detach().cpu().numpy()
    frwiki_dict[key] = fr.embedding.detach().cpu().numpy()
    dewiki_dict[key] = de.embedding.detach().cpu().numpy()
print('Done!')

# write train data
print('Writing data into HDF5 files...')
for key_label in enwiki_dict.keys():
    with h5py.File(os.path.join(HDF5_DIR, f'{key_label}_pilot.h5'), 'w') as h5f:
        for i, key in enumerate(enwiki_dict[key_label]):

            # save values
            h5f.create_dataset(name=f'{key.strip("Q")}_enwiki', data=enwiki_dict[key], compression="gzip", compression_opts=9)
            h5f.create_dataset(name=f'{key.strip("Q")}_frwiki', data=frwiki_dict[key], compression="gzip", compression_opts=9)
            h5f.create_dataset(name=f'{key.strip("Q")}_dewiki', data=dewiki_dict[key], compression="gzip", compression_opts=9)

        # save keys as int
        h5f.create_dataset(name=f'ids', data=np.array([key.strip('Q') for key in enwiki_dict[key_label]], dtype=np.int), compression="gzip", compression_opts=9)
print('Done!')

tok = dt.datetime.now()

print(f'Total time: {tok-tik} seconds')
