import os
import random
import torch
import numpy as np
from pathlib import Path
from args import get_parser

ROOT_PATH = Path(os.path.dirname(__file__))

# read parser
parser = get_parser()
args = parser.parse_args()

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
def rank(opts, img_embeds, text_embeds, ids):
    type_embedding = opts.embtype
    im_vecs = img_embeds
    instr_vecs = text_embeds
    names = ids

    # Sort based on names to always pick same samples for medr
    idxs = np.argsort(names)
    names = names[idxs]

    # Ranker
    N = opts.medr
    idxs = range(N)

    glob_rank = []
    glob_recall = {1:0.0, 5:0.0, 10:0.0}

    for i in range(10):
        ids = random.sample(range(0,len(names)), N)
        im_sub = im_vecs[ids,:]
        instr_sub = instr_vecs[ids,:]
        ids_sub = names[ids]

        if type_embedding == 'image':
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
        # print "median", med

        for i in recall.keys():
            glob_recall[i]+=recall[i]
        glob_rank.append(med)

    for i in glob_recall.keys():
        glob_recall[i] = format(glob_recall[i]/10, '.4f')

    return np.average(glob_rank), glob_recall

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    filename = f'{ROOT_PATH}/{args.snapshots}/model_e{state["epoch"]}_v-{state["best_val"]:.4f}.pth.tar'
    torch.save(state, filename)

def adjust_learning_rate(optimizer, epoch, opts):
    """Switching between modalities"""
    # parameters corresponding to the rest of the network
    optimizer.param_groups[0]['lr'] = opts.lr * opts.freeText
    # parameters corresponding to visionMLP
    optimizer.param_groups[1]['lr'] = opts.lr * opts.freeVision

    print(f'Initial base params lr: {optimizer.param_groups[0]["lr"]}')
    print(f'Initial vision lr: {optimizer.param_groups[1]["lr"]}')

    # after first modality change we set patience to 3
    opts.patience = 3