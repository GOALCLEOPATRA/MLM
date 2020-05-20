import os
import time
import random
import logging
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from args import get_parser
from data.data_loader import MLMLoader
from models.model import MLMBaseline
from utils import IRLoss, LELoss, MTLLoss, AverageMeter, rank, classify

# define criterions
criterions = {
    'ir': IRLoss,
    'le': LELoss,
    'mtl': MTLLoss
}

ROOT_PATH = Path(os.path.dirname(__file__))

# read parser
parser = get_parser()
args = parser.parse_args()

# create directories for train experiments
logging_path = f'{args.path_results}/{args.data_path.split("/")[-1]}/{args.task}'
Path(logging_path).mkdir(parents=True, exist_ok=True)

# set logger
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(f'{logging_path}/train.log', 'w'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

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
    # set model
    model = MLMBaseline()
    model.to(device)

    # define loss function
    criterion = criterions[args.task]()

    logger.info(f"=> loading checkpoint '{args.model_path}'")
    if device.type == 'cpu':
        checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1', map_location='cpu')
    else:
        checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1')
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f"=> loaded checkpoint '{args.model_path}' (epoch {checkpoint['epoch']})")

    # prepare test loader
    test_loader = torch.utils.data.DataLoader(
        MLMLoader(data_path=f'{ROOT_PATH}/{args.data_path}', partition='test'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    logger.info('Test loader prepared.')

    # run test
    test(test_loader, model, criterion)

def test(val_loader, model, criterion):
    losses = {
        'ir': AverageMeter(),
        'le': AverageMeter(),
        'mtl': AverageMeter()
    }
    le_img = []
    le_txt = []

    # switch to evaluate mode
    model.eval()

    for i, val_input in enumerate(val_loader):
        # inputs
        images = torch.stack([val_input['image'][j].to(device) for j in range(len(val_input['image']))])
        summaries = torch.stack([val_input['summary'][j].to(device) for j in range(len(val_input['summary']))])
        triples = torch.stack([val_input['triple'][j].to(device) for j in range(len(val_input['triple']))])

        # target
        target = {
            'ir': torch.stack([val_input['target_ir'][j].to(device) for j in range(len(val_input['target_ir']))]),
            'le': torch.stack([val_input['target_le'][j].to(device) for j in range(len(val_input['target_le']))]),
            'ids': torch.stack([val_input['id'][j].to(device) for j in range(len(val_input['id']))])
        }

        # compute output
        output = model(images, summaries, triples)

        # compute loss
        loss = criterion(output, target)

        # measure performance and record loss
        if args.task == 'mtl':
            losses['mtl'].update(loss['mtl'].data, args.batch_size)
            losses['ir'].update(loss['ir'].data, args.batch_size)
            losses['le'].update(loss['le'].data, args.batch_size)
            log_loss = f'IR: {losses["ir"].val:.4f} ({losses["ir"].avg:.4f}) - LE: {losses["le"].val:.4f} ({losses["le"].avg:.4f})'
        else:
            losses[args.task].update(loss.data, args.batch_size)
            log_loss = f'{losses[args.task].val:.4f} ({losses[args.task].avg:.4f})'

        if args.task in ['ir', 'mtl']:
            if i==0:
                data0 = output['ir'][0].data.cpu().numpy()
                data1 = output['ir'][1].data.cpu().numpy()
                data2 = target['ids'].data.cpu().numpy()
            else:
                data0 = np.concatenate((data0, output['ir'][0].data.cpu().numpy()), axis=0)
                data1 = np.concatenate((data1, output['ir'][1].data.cpu().numpy()), axis=0)
                data2 = np.concatenate((data2, target['ids'].data.cpu().numpy()), axis=0)

        if args.task in ['le', 'mtl']:
            le_img.append([[t, torch.topk(o, k=1)[1], torch.topk(o, k=5)[1], torch.topk(o, k=10)[1]] for o, t in zip(output['le'][0], target['le'])])
            le_txt.append([[t, torch.topk(o, k=1)[1], torch.topk(o, k=5)[1], torch.topk(o, k=10)[1]] for o, t in zip(output['le'][1], target['le'])])


    results = {
        'log': {
            'Loss': log_loss
        }
    }

    if args.task in ['ir', 'mtl']:
        rank_results = rank(data0, data1, data2)
        results['log']['IR Median Rank'] = rank_results['median_rank']
        results['log']['IR Precision'] = ' - '.join([f'{k}: {v}' for k, v in rank_results['precision'].items()])
        results['log']['IR Recall'] = ' - '.join([f'{k}: {v}' for k, v in rank_results['recall'].items()])
        results['log']['IR F1 score'] = ' - '.join([f'{k}: {v}' for k, v in rank_results['f1_score'].items()])
        results['log']['IR Mean'] = ' - '.join([f'{k}: {v}' for k, v in rank_results['mean'].items()])

    if args.task in ['le', 'mtl']:
        classify_results = classify(le_img, le_txt)
        results['log']['LE Image'] = ' - '.join([f'{k}: {v}' for k, v in classify_results['image'].items()])
        results['log']['LE Text'] = ' - '.join([f'{k}: {v}' for k, v in classify_results['text'].items()])

    # log results
    for k, v in results['log'].items():
        logger.info(f'** Test {k} - {v}')

if __name__ == '__main__':
    main()