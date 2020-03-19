


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as nps
from torch.autograd import Variable
import glob



# Import ResNet-152
resnet152 = models.resnet152(pretrained=True)
modules=list(resnet152.children())[:-1]
resnet152=nn.Sequential(*modules)
for p in resnet152.parameters():
    p.requires_grad = False

# Img transforms to fit ResNet
scaler = transforms.Resize((224, 224))
to_tensor = transforms.ToTensor()

# Open images and 
embs = []
images = glob.glob("thumbnails/*") # add path
for image in images:
    with open(image, 'rb') as file:
        img = Image.open(file)
        img = Variable(to_tensor(scaler(img)).unsqueeze(0))
        emb_var = resnet152(img) # embeddings from last layer
        emb = emb_var.data
        emb = emb.view(2048)
        print('emb', emb, emb.size())
        embs.append(emb)

print('embs', len(embs))