'''
Create one HDF5 file and store image embeddings
'''
import os
import h5py
import json
import torch
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

# set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
images = glob(os.path.join(SOURCE, 'thumbnails/*')) # for now we take all images

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

# write train data
print('Writing data into HDF5 files...')
for key_label in img_dict.keys():
    with h5py.File(os.path.join(HDF5_DIR, f'{key_label}_pilot.h5'), 'w') as h5f:
        for i, key in enumerate(img_dict[key_label]):
            # create image matrix
            image_matrix = np.stack(img_dict[key])

        # save keys as int
        h5f.create_dataset(name=f'ids', data=np.array([key.strip('Q') for key in img_dict[key_label]], dtype=np.int), compression="gzip", compression_opts=9)
print('Done!')

tok = dt.datetime.now()

print(f'Total time: {tok-tik} seconds')