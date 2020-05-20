import os
import random
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from visdom import Visdom
from args import get_parser

# read parser
parser = get_parser()
args = parser.parse_args()

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# meter class for storing results
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# ranking method for evaluating results
def rank(img_embeds, text_embeds, names):
    # Sort based on names to always pick same samples for medr
    idxs = np.argsort(names)
    names = names[idxs]

    # Ranker
    N = args.medr
    idxs = range(N)

    glob_rank = []
    glob_recall = {1:0.0, 5:0.0, 10:0.0}

    for i in range(10):
        ids = random.sample(range(0,len(names)), N)
        im_sub = img_embeds[ids,:]
        instr_sub = text_embeds[ids,:]
        ids_sub = names[ids]

        if args.emb_type == 'image':
            sims = np.dot(im_sub, instr_sub.T) # for im2text
        else:
            sims = np.dot(instr_sub, im_sub.T) # for text2im

        med_rank = []
        recall = {1:0.0, 5:0.0, 10:0.0}

        for ii in idxs:
            name = ids_sub[ii]
            # get a column of similarities
            sim = sims[ii, :]

            # sort indices in descending order
            sorting = np.argsort(sim)[::-1].tolist()

            # find where the index of the pair sample ended up in the sorting
            pos = sorting.index(ii)

            if (pos+1) == 1:
                recall[1]+=1
            if (pos+1) <=5:
                recall[5]+=1
            if (pos+1)<=10:
                recall[10]+=1

            # store the position
            med_rank.append(pos+1)

        for i in recall.keys():
            recall[i]=recall[i]/N

        med = np.median(med_rank)

        for i in recall.keys():
            glob_recall[i]+=recall[i]
        glob_rank.append(med)

    for i in glob_recall.keys():
        glob_recall[i] = format(glob_recall[i]/10, '.4f')

    return np.average(glob_rank), glob_recall

def save_checkpoint(state, path):
    filename = f'{path}/epoch_{state["epoch"]}_loss_{state["val_loss"]:.2f}.pth.tar'
    torch.save(state, filename)

class IRLoss(nn.Module):
    '''Information Retrieval Loss'''
    def __init__(self):
        super().__init__()
        self.criterion = nn.CosineEmbeddingLoss(margin=0.1).to(device)

    def forward(self, output, target):
        return self.criterion(output['ir'][0], output['ir'][1], target['ir'])

class LELoss(nn.Module):
    '''Location Estimation Loss'''
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.criterion(output['le'][0], target['le']) + self.criterion(output['le'][1], target['le'])

class MTLLoss(nn.Module):
    '''Multi Task Learning Loss'''
    def __init__(self):
        super().__init__()
        self.ir_loss = IRLoss()
        self.le_loss = LELoss()

        self.mml_emp = torch.Tensor([True, False])
        self.log_vars = torch.nn.Parameter(torch.zeros(len(self.mml_emp)))

    def forward(self, output, target):
        task_losses = torch.stack((self.ir_loss(output, target), self.le_loss(output, target)))
        dtype = task_losses.dtype

        # weighted loss
        stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)
        weights = 1 / ((self.mml_emp.to(device).to(dtype)+1)*(stds**2))
        losses = weights * task_losses + torch.log(stds)

        return {
            'ir': losses[0],
            'le': losses[1],
            'mtl': losses.mean()
        }

# visualisations using visdom
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))

        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')