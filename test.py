import os
import time
import random
import pickle
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from args import get_parser
from utils import AverageMeter, rank
from data.data_loader import MLMLoader
from models.model import MLMCoordPrediction

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

    model = MLMCoordPrediction()
    model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.BCEWithLogitsLoss()

    print(f"=> loading checkpoint '{args.model_path}'")
    if device.type=='cpu':
        checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1', map_location='cpu')
    else:
        checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1')
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print(f"=> loaded checkpoint '{args.model_path}' (epoch {checkpoint['epoch']})")

    # prepared validation loader
    val_loader = torch.utils.data.DataLoader(
        MLMLoader(data_path=f'{ROOT_PATH}/{args.data_path}', partition='val'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    print('Validation loader prepared.')

    # prepare test loader
    test_loader = torch.utils.data.DataLoader(
        MLMLoader(data_path=f'{ROOT_PATH}/{args.data_path}', partition='test'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    print('Test loader prepared.')

    # run test
    test(val_loader, model, criterion)
    test(test_loader, model, criterion)

def test(test_loader, model, criterion):
    img_losses = AverageMeter()
    txt_losses = AverageMeter()
    pred_img = []
    pred_txt = []
    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, val_input in enumerate(test_loader):
        # inputs
        image = torch.stack([val_input['image'][j].to(device) for j in range(len(val_input['image']))])
        text = torch.stack([val_input['multi_wiki'][j].to(device) for j in range(len(val_input['multi_wiki']))])

        # target
        coord_target = torch.stack([val_input['coord'][j].to(device) for j in range(len(val_input['coord']))])

        # compute output
        img_coord, txt_coord = model(image, text)

        # compute loss
        img_loss = criterion(img_coord, coord_target)
        txt_loss = criterion(txt_coord, coord_target)

        # measure performance and record loss
        img_losses.update(img_loss.data, image.size(0))
        txt_losses.update(txt_loss.data, text.size(0))

        # predict
        pred_img.append([[t.argmax(), torch.topk(o, k=1)[1], torch.topk(o, k=5)[1], torch.topk(o, k=10)[1]] for o, t in zip(img_coord, coord_target)])
        pred_txt.append([[t.argmax(), torch.topk(o, k=1)[1], torch.topk(o, k=5)[1], torch.topk(o, k=10)[1]] for o, t in zip(txt_coord, coord_target)])

    pred_img = [p for pred in pred_img for p in pred]
    pred_txt = [p for pred in pred_txt for p in pred]

    print(f'Image Loss: {img_losses.avg}')
    print(f'==>Pred@1: {sum([1 if p[0] in p[1] else 0 for p in pred_img])/len(pred_img):.2f}')
    print(f'==>Pred@5: {sum([1 if p[0] in p[2] else 0 for p in pred_img])/len(pred_img):.2f}')
    print(f'==>Pred@10: {sum([1 if p[0] in p[3] else 0 for p in pred_img])/len(pred_img):.2f}')
    print(f'Text Loss: {txt_losses.avg}')
    print(f'==>Pred@1: {sum([1 if p[0] in p[1] else 0 for p in pred_txt])/len(pred_txt):.2f}')
    print(f'==>Pred@5: {sum([1 if p[0] in p[2] else 0 for p in pred_txt])/len(pred_txt):.2f}')
    print(f'==>Pred@10: {sum([1 if p[0] in p[3] else 0 for p in pred_txt])/len(pred_txt):.2f}')


if __name__ == '__main__':
    main()