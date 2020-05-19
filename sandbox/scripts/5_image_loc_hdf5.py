'''
Create HDF5 file with image resnet embeddings
'''
import os
import h5py
import json
import time
import torch
import random
import numpy as np
import torch.nn as nn
from glob import glob
from PIL import Image
from pathlib import Path
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
from location_verification.location_embedding import GeoEstimator

# set
torch.cuda.set_device(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent.parent

# load data
data = []
for i in range(1, 21):
    data_path = f'{str(ROOT_PATH)}/mlm_dataset/MLM_dataset/MLM_{i}/MLM_{i}.json'
    with open(data_path, encoding='utf-8') as json_file:
        data.extend(json.load(json_file))
ids = set([d['id'] for d in data])

# load images
images = []
for i in range(1, 21):
    images.extend(glob(f'{str(ROOT_PATH)}/mlm_dataset/MLM_dataset/MLM_{i}/images_{i}/*'))

# Import ResNet-152
# print('Loading ResNet-152...')
# resnet152 = models.resnet152(pretrained=True)
# modules = list(resnet152.children())[:-1]
# resnet152 = nn.Sequential(*modules).to(DEVICE)
# for p in resnet152.parameters():
#     p.requires_grad = False
# print('Done!')
model_path_loc = 'C:/Users/TahmasebzadehG/PycharmProjects/InfoRetrieval/resources/geolocation_estimation/base_M/'
location_obj = GeoEstimator(model_path_loc, use_cpu=True)


# Img transforms to fit ResNet
scaler = transforms.Resize((224, 224))
to_tensor = transforms.ToTensor()

# Create image dictionary
print('Processing images...')
img_dict = {}
for img in images:
    img_name = img.rsplit("\\", 1)[-1]
    key = int(img_name.split('_')[0].strip('Q'))
    if key in ids:
        if key not in img_dict:
            img_dict[key] = []

        img_dict[key].append(img)
print('Finished processing images...')

# write train data
HDF5_DIR = ROOT_PATH / f'mlm_dataset/hdf5/images/images_loc.h5'
print('Getting ResNet embeddings...')
tic = time.perf_counter()
with h5py.File(HDF5_DIR, 'a') as h5f:
    for i, id in enumerate(list(img_dict.keys())):
        print(f'Getting results for id {id}')
        if f'{id}_images' in h5f:
            print(f'====>ID {id} already exists in the dataset!')
            continue
        outputs = []
        for img in img_dict[id]:
            im_pil = Image.open(img).convert('RGB')
            outputs.append(Variable(to_tensor(scaler(im_pil)).to(DEVICE).unsqueeze(0)))

        result = torch.cat(outputs, dim=0)
        # emb_var = resnet152(result) # embeddings from last layer
        # emb = emb_var.data.view(result.size(0), 2048).detach().cpu().numpy()
        location_features = location_obj.get_img_embedding(img)
        # save values
        h5f.create_dataset(name=f'{id}_images_loc', data=location_features, compression="gzip", compression_opts=9)
        toc = time.perf_counter()
        print(f'====> Finished images for id {id} -- {((i+1)/len(list(img_dict.keys())))*100:.2f}% -- {toc - tic:0.2f}s')

    # save keys as int
    h5f.create_dataset(name=f'ids', data=np.array(list(img_dict.keys()), dtype=np.int), compression="gzip", compression_opts=9)
print('Done!')