'''
Create one HDF5 file and store whole dataset

Stats
3.21 minutes for 298 examples
HDF5:
    - All images: 1.6 GB for images, wiki texts (en, fr, de) and coords - triples are not included
    - One image:
Deepdish:
    - All images: 224MB for all modalities and images (including triples)
    - One image: 71MB
'''

import os
import h5py
import json
import torch
import flair
import random
import numpy as np
from pathlib import Path
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import datetime as dt
from glob import glob
from PIL import Image
from flair.embeddings import FlairEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence
from sklearn.model_selection import train_test_split

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# set device
flair.device = DEVICE

torch.cuda.device(1)

tik = dt.datetime.now()

# Define contstants
HEIGHT = 256
WIDTH = 256
CHANNELS = 3
EMBED_SIZE = 2048
COORD_SIZE = 2

# Define paths
ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent.parent
HDF5_DIR = ROOT_PATH / 'dataset'
SOURCE = ROOT_PATH / 'dataset_files/pilot/'
data_path = os.path.join(SOURCE, 'capitals_final.json')
wikidata_path = os.path.join(SOURCE, 'capitals_wikidata.json')
images = glob(os.path.join(SOURCE, 'thumbnails/*')) # for now we take all images

# Read data from json
data = []
with open(data_path) as json_file:
    data = json.load(json_file)

wikidata = []
with open(wikidata_path) as json_file:
    wikidata = json.load(json_file)


# Import ResNet-152
print('Loading ResNet-152...')
resnet152 = models.resnet152(pretrained=True)
modules = list(resnet152.children())[:-1]
resnet152 = nn.Sequential(*modules)
for p in resnet152.parameters():
    p.requires_grad = False
print('Done!')

# Img transforms to fit ResNet
scaler = transforms.Resize((224, 224))
to_tensor = transforms.ToTensor()

# Create image dictionary
print('Processing with images...')
img_dict = {}
for img in images:
    key = img.rsplit('/', 1)[-1].split('_')[0]
    if key not in img_dict:
        img_dict[key] = []

    img = Image.open(img).convert('RGB')
    img = Variable(to_tensor(scaler(img)).unsqueeze(0))
    emb_var = resnet152(img) # embeddings from last layer
    emb = emb_var.data
    emb = emb.view(2048).detach().cpu().numpy()
    img_dict[key].append(emb)

print('Done!')


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
coord_dict = {}
for sample in data:
    key = sample['item'].rsplit('/', 1)[-1]
    # Get values
    en = Sentence(sample['en_wiki']['text'])
    fr = Sentence(sample['fr_wiki']['text'])
    de = Sentence(sample['de_wiki']['text'])
    coord = sample['coord'][sample['coord'].find("(")+1:sample['coord'].find(")")].split(' ')

    # Process values
    document_embeddings.embed(en)
    document_embeddings.embed(fr)
    document_embeddings.embed(de)
    coord = np.array(coord, dtype=np.float32)

    # Save values
    enwiki_dict[key] = en.embedding.detach().cpu().numpy()
    frwiki_dict[key] = fr.embedding.detach().cpu().numpy()
    dewiki_dict[key] = de.embedding.detach().cpu().numpy()
    coord_dict[key] = coord
print('Done!')

# Create triple dictionary
# print('Processing with triples...')
# We take only english for now - We do triples later
# triple_dict = {}
# for sample in wikidata:
#     key = sample['item'].rsplit('/', 1)[-1]
#     if key in enwiki_dict:
#         triple_dict[key] = sample['wikidata']['en']
# print('Done!')

# assert len(img_dict) == len(enwiki_dict), 'Images and data does not have the same size'

all_ids = list(img_dict.keys())
train_ids, test_ids = train_test_split(all_ids, test_size=0.2, shuffle=True)
test_ids, val_ids = train_test_split(test_ids, test_size=0.5)

dict_ids = {
    'train': train_ids,
    'val': val_ids,
    'test': test_ids
}

# write train data
print('Writing data into HDF5 files...')
for key_label in dict_ids.keys():
    with h5py.File(os.path.join(HDF5_DIR, f'{key_label}_pilot.h5'), 'w') as h5f:
        for i, key in enumerate(dict_ids[key_label]):
            # create image matrix
            image_matrix = np.stack(img_dict[key])

            # save values
            h5f.create_dataset(name=f'{key.strip("Q")}_images', data=image_matrix, compression="gzip", compression_opts=9)
            h5f.create_dataset(name=f'{key.strip("Q")}_enwiki', data=enwiki_dict[key], compression="gzip", compression_opts=9)
            h5f.create_dataset(name=f'{key.strip("Q")}_frwiki', data=frwiki_dict[key], compression="gzip", compression_opts=9)
            h5f.create_dataset(name=f'{key.strip("Q")}_dewiki', data=dewiki_dict[key], compression="gzip", compression_opts=9)
            h5f.create_dataset(name=f'{key.strip("Q")}_coords', data=coord_dict[key], compression="gzip", compression_opts=9)
            # h5f.create_dataset(name=f'{key.strip("Q")}_triples', data=triple_dict[key], compression="gzip", compression_opts=9)

        # save keys as int
        h5f.create_dataset(name=f'ids', data=np.array([key.strip('Q') for key in dict_ids[key_label]], dtype=np.int), compression="gzip", compression_opts=9)
print('Done!')

tok = dt.datetime.now()

print(f'Total time: {tok-tik} seconds')

# mlm_pilot = {
#     'images': img_dict,
#     'en_wikis': enwiki_dict,
#     'fr_wikis': frwiki_dict,
#     'de_wikis': dewiki_dict,
#     'coords': coord_dict,
#     'triples': triple_dict
# }

# import deepdish as dd
# dd.io.save(os.path.join(HDF5_DIR, 'mlm_pilot.h5'), mlm_pilot)
# test = dd.io.load(os.path.join(HDF5_DIR, 'mlm_pilot.h5'))
# tok = dt.datetime.now()
# print(f'{(tok-tik).seconds} seconds')