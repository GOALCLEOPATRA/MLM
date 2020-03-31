import os
import random
import torch
import numpy as np
from pathlib import Path
from args import get_parser
from  more_itertools import unique_everseen

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
def rank(opts, input_embeds, coord_embeds, ids):
    im_vecs = input_embeds
    coord_vecs = coord_embeds
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
        input_sub = im_vecs[ids,:]
        coord_sub = coord_vecs[ids,:]
        ids_sub = names[ids]

        sims = np.dot(input_sub, coord_sub.T) # for input2coord

        med_rank = []
        recall = {1:0.0, 5:0.0, 10:0.0}

        for ii in idxs:
            name = ids_sub[ii]
            # get a column of similarities
            sim = sims[ii, :]

            # sort indices in descending order
            sorting = np.argsort(sim)[::-1].tolist()

            # we want unique index since we use coordinates ids
            sorting = ids_sub[sorting].tolist()
            sorting = list(unique_everseen(sorting))

            # find where the index of the pair sample ended up in the sorting
            pos = sorting.index(name)

            if (pos+1) == 1: recall[1] += 1
            if (pos+1) <= 5: recall[5] += 1
            if (pos+1) <= 10: recall[10] += 1

            # store the position
            med_rank.append(pos+1)

        unique_coord_num = len(list(unique_everseen(ids_sub)))
        for i in recall.keys():
            recall[i]=recall[i]/unique_coord_num

        for i in recall.keys():
            glob_recall[i]+=recall[i]
        glob_rank.append(np.median(med_rank))

    for i in glob_recall.keys():
        glob_recall[i] = format(glob_recall[i]/10, '.2f')

    return np.average(glob_rank), glob_recall

def save_checkpoint(state, input_name):
    filename = f'{ROOT_PATH}/{args.snapshots}/{input_name}_model_e{state["epoch"]}_v-{state["best_val"]:.2f}.pth.tar'
    torch.save(state, filename)