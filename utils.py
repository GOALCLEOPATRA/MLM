import os
import random
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from visdom import Visdom
from args import get_parser
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    glob_recall = {i+1: 0.0 for i in range(10)}
    glob_precision = {i+1: 0.0 for i in range(10)}

    for i in range(args.rank_times):
        ids = random.sample(range(0,len(names)), N)
        im_sub = img_embeds[ids,:]
        instr_sub = text_embeds[ids,:]
        ids_sub = names[ids]

        if args.emb_type == 'image':
            sims = np.dot(im_sub, instr_sub.T) # for im2text
        else:
            sims = np.dot(instr_sub, im_sub.T) # for text2im

        med_rank = []
        recall = {i+1: 0.0 for i in range(10)}
        precision = {i+1: 0.0 for i in range(10)}

        for ii in idxs:
            name = ids_sub[ii]
            # get a column of similarities
            sim = sims[ii, :]

            # sort indices in descending order
            sorting = np.argsort(sim)[::-1].tolist()

            # find where the index of the pair sample ended up in the sorting
            pos = sorting.index(ii)

            # store the position
            med_rank.append(pos+1)

            # recall
            for k in precision.keys():
                if (pos+1) <= k:
                    recall[k] += 1

            # precision - we consider that we retrieve 10 samples
            relevance = [1 if i == pos else 0 for i in range(10)]
            for k in precision.keys():
                precision[k] += np.mean((np.asarray(relevance)[:k] != 0))

        # save median rank for every run
        glob_rank.append(np.median(med_rank))

        # update recall for every run
        for i in recall.keys():
            recall[i] = recall[i]/N
            glob_recall[i] += recall[i]

        # update precision for every run
        for k in precision.keys():
            precision[k] = precision[k]/N
            glob_precision[k] += precision[k]

    # calculate final recall values
    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i]/args.rank_times

    # calculate final precision values
    for k in precision.keys():
        glob_precision[k] = glob_precision[k]/args.rank_times

    # calculate ranking metrics
    mean_precision = np.mean(np.asarray(list(glob_precision.values())).astype(np.float))
    mean_recall = np.mean(np.asarray(list(glob_recall.values())).astype(np.float))
    mean_f1score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)

    return {
        'median_rank': np.average(glob_rank),
        'precision': {
            'P@1': format(glob_precision[1], '.4f'),
            'P@5': format(glob_precision[5], '.4f'),
            'P@10': format(glob_precision[10], '.4f'),
        },
        'recall': {
            'R@1': format(glob_recall[1], '.4f'),
            'R@5': format(glob_recall[5], '.4f'),
            'R@10': format(glob_recall[10], '.4f')
        },
        'f1_score': {
            'F1@1':  format(2 * (glob_precision[1] * glob_recall[1]) / (glob_precision[1] + glob_recall[1]), '.4f'),
            'F1@5':  format(2 * (glob_precision[5] * glob_recall[5]) / (glob_precision[5] + glob_recall[5]), '.4f'),
            'F1@10': format(2 * (glob_precision[10] * glob_recall[10]) / (glob_precision[10] + glob_recall[10]), '.4f')
        },
        'mean': {
            'Precision': format(mean_precision, '.4f'),
            'Recall': format(mean_recall, '.4f'),
            'F1_score': format(mean_f1score, '.4f')
        }
    }

def classify(le_img, le_txt):
    # flatten lists
    le_img = [p for pred in le_img for p in pred]
    le_txt = [p for pred in le_txt for p in pred]

    # extract targets and top predictions for both image and text
    trg_img = [p[0].item() for p in le_img]
    top1_img = [p[1].item() for p in le_img]
    trg_txt = [p[0].item() for p in le_txt]
    top1_txt = [p[1].item() for p in le_txt]

    return {
        'image': {
            'Accuracy': format(accuracy_score(trg_img, top1_img), '.4f'),
            'Precision': format(precision_score(trg_img, top1_img, average='weighted'), '.4f'),
            'Recall': format(recall_score(trg_img, top1_img, average='weighted'), '.4f'),
            'F1 score': format(f1_score(trg_img, top1_img, average='weighted'), '.4f')
        },
        'text': {
            'Accuracy': format(accuracy_score(trg_txt, top1_txt), '.4f'),
            'Precision': format(precision_score(trg_txt, top1_txt, average='weighted'), '.4f'),
            'Recall': format(recall_score(trg_txt, top1_txt, average='weighted'), '.4f'),
            'F1 score': format(f1_score(trg_txt, top1_txt, average='weighted'), '.4f')
        }
    }

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