import os
import time
import random
import pickle
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from args import get_parser
from data.data_loader_coord_cluster import MLMLoader
from models.model_t2 import MLMRetrieval
from utils import AverageMeter, rank

ROOT_PATH = Path(os.path.dirname(__file__))


# read parser
parser = get_parser()
args = parser.parse_args()

# set a seed value
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    model = MLMRetrieval()
    model.to(device)

    # define loss function (criterion) and optimizer
    # bce for classification
    criterion_t2 = nn.CrossEntropyLoss()

    print(f"=> loading checkpoint '{args.model_path}'")
    if device.type=='cpu':
        checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1', map_location='cpu')
    else:
        checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1')
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print(f"=> loaded checkpoint '{args.model_path}' (epoch {checkpoint['epoch']})")

    # prepare test loader
    test_loader = torch.utils.data.DataLoader(
        MLMLoader(data_path=f'{ROOT_PATH}/{args.data_path}', partition='test'),
        # batch_size=args.batch_size,
        batch_size=1000,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    print('Test loader prepared.')

    # run test
    kff(test_loader, model, criterion_t2)

def kff(test_loader, model, criterion_t2):
# def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    t2_losses = AverageMeter()
    pred_t2_img = []
    pred_t2_txt = []

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, test_input in enumerate(test_loader):
                # inputs

        input_img = torch.stack([test_input['image'][j].to(device) for j in range(len(test_input['image']))])
        input_summary = torch.stack([test_input['multi_wiki'][j].to(device) for j in range(len(test_input['multi_wiki']))])
        input_tri =  torch.stack([test_input['triple'][j].to(device) for j in range(len(test_input['triple']))])

        # target
        cluster_target = torch.stack([test_input['cluster'][j].to(device) for j in range(len(test_input['cluster']))])

        cluster_target_idx = cluster_target.long().argmax(1)


        # compute output

        output = model(input_img, input_summary, input_tri)
        # compute loss
        m_loss_t2_img = criterion_t2(output[0], cluster_target_idx)
        m_loss_t2_txt = criterion_t2(output[2], cluster_target_idx)
        m_loss_t2 = m_loss_t2_img + m_loss_t2_txt
        # measure performance and record loss
        t2_losses.update(m_loss_t2.data, input_img.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        
        pred_t2_img.append([[t.argmax(), torch.topk(o, k=1)[1], torch.topk(o, k=5)[1], torch.topk(o, k=10)[1]] for o, t in zip(output[0], cluster_target)])
        pred_t2_txt.append([[t.argmax(), torch.topk(o, k=1)[1], torch.topk(o, k=5)[1], torch.topk(o, k=10)[1]] for o, t in zip(output[2], cluster_target)])
    pred_t2_img = [p for pred in pred_t2_img for p in pred]
    pred_t2_txt = [p for pred in pred_t2_txt for p in pred]

    print(f'Task 2 Loss: {"%.4f" % t2_losses.avg}')
    print(f'Image Predictions')
    print(f'==>Pred@1: {sum([1 if p[0] in p[1] else 0 for p in pred_t2_img])/len(pred_t2_img):.2f}')
    print(f'==>Pred@5: {sum([1 if p[0] in p[2] else 0 for p in pred_t2_img])/len(pred_t2_img):.2f}')
    print(f'==>Pred@10: {sum([1 if p[0] in p[3] else 0 for p in pred_t2_img])/len(pred_t2_img):.2f}')
    print(f'Text Predictions')
    print(f'==>Pred@1: {sum([1 if p[0] in p[1] else 0 for p in pred_t2_txt])/len(pred_t2_txt):.2f}')
    print(f'==>Pred@5: {sum([1 if p[0] in p[2] else 0 for p in pred_t2_txt])/len(pred_t2_txt):.2f}')
    print(f'==>Pred@10: {sum([1 if p[0] in p[3] else 0 for p in pred_t2_txt])/len(pred_t2_txt):.2f}')

if __name__ == '__main__':
    main()