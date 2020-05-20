




''' Read in embs from h5py '''

# imports

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from torch import autograd, nn, tanh, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.functional import softplus
from torch.nn import BCEWithLogitsLoss
from torchvision import transforms
import h5py
import numpy as np 
from pathlib import Path


# parameters

batch_size = 2
textual_dim = 3072
visual_dim = 3072
out_dim = 512
mm_dim1 = 512
mm_dim2 = 512
out_dim1 = 512
out_dim2 = 512
num_classes = 6
learning_rate = 0.1
num_epochs = 2
dataload_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0}


# paths
data_path = 'C:/Users/JasonA1/Documents/pythonscripts/wiki-mlm/pilot'
text_file = '/train_embs_text.hdf5'
#image_file = '/train_image.hdf5'

# dataset class
class HDF5Data_t(Dataset):
    def __init__(self, hdf5file, embs_key, labels_key,
         transform=None):
         
        self.hdf5file=hdf5file
        self.embs_key=embs_key
        self.labels_key=labels_key
        self.transform=transform

    def __len__(self):
        with h5py.File(self.hdf5file, 'r') as db:
            lens=len(db[self.labels_key])
        return lens
    
    def __getitem__(self, idx):
        with h5py.File(self.hdf5file,'r') as db:
            x_text=db[self.embs_key][idx]
            y_text=db[self.labels_key][idx]
            sample={'x_text':x_text,'y_text':y_text}
        if self.transform:
            sample=self.transform(sample)
        return sample 

# dataset call
dataset_t = HDF5Data_t(data_path+text_file, 'bert_embs', 'entity_ids') # specify input file, data, labels

# dataloader object
train_loader_t = DataLoader(dataset_t, **dataload_params)

# print items from dataloader
for i in train_loader_t:
    print('multi', i)
