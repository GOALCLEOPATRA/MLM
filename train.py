import os
import time
import random
import torch
import logging
import numpy as np
import torch.nn as nn
from pathlib import Path
from args import get_parser
from models.model import MLMBaseline
from data.data_loader import MLMLoader
from utils import IRLoss, LELoss, MTLLoss, AverageMeter, rank, save_checkpoint

# define models
# models = {
#     'ir': MLMBaseline,
#     'le': MLMBaseline,
#     'mtl': MLMBaseline
# }

# define criterions
criterions = {
    'ir': IRLoss,
    'le': LELoss,
    'mtl': MTLLoss
}

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

# read parser
parser = get_parser()
args = parser.parse_args()

# create directories for train experiments
logging_path = f'{args.path_results}/{args.data_path.split("/")[-1]}/{args.task}'
Path(logging_path).mkdir(parents=True, exist_ok=True)
checkpoint_path = f'{args.snapshots}/{args.data_path.split("/")[-1]}/{args.task}'
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

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

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # set model
    model = MLMBaseline()
    model.to(device)

    # define loss function and optimizer
    criterion = criterions[args.task]()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # load checkpoint
    if os.path.isfile(args.resume):
        logger.info(f"=> loading checkpoint '{args.resume}''")
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

    # log num of params
    logger.info(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    # prepare training loader
    train_loader = torch.utils.data.DataLoader(
        MLMLoader(data_path=f'{ROOT_PATH}/{args.data_path}', partition='train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    logger.info('Training loader prepared.')

    # prepared validation loader
    val_loader = torch.utils.data.DataLoader(
        MLMLoader(data_path=f'{ROOT_PATH}/{args.data_path}', partition='val'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    logger.info('Validation loader prepared.')

    # run epochs
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch+1) % args.valfreq == 0:
            val_result = validate(val_loader, model, criterion)

            # save the best model
            save_checkpoint({
                'data': args.data_path.split('/')[-1],
                'task': args.task,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_result['loss']},
                path=checkpoint_path)

            for k, v in val_result['log'].items():
                logger.info(f'** Val {k} - {v}')


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = {
        'ir': AverageMeter(),
        'le': AverageMeter(),
        'mtl': AverageMeter()
    }

    # switch to train mode
    model.train()

    end = time.time()
    for i, train_input in enumerate(train_loader):
        # inputs
        images = torch.stack([train_input['image'][j].to(device) for j in range(len(train_input['image']))])
        summaries = torch.stack([train_input['summary'][j].to(device) for j in range(len(train_input['summary']))])
        triples = torch.stack([train_input['triple'][j].to(device) for j in range(len(train_input['triple']))])

        # target
        target = {
            'ir': torch.stack([train_input['target_ir'][j].to(device) for j in range(len(train_input['target_ir']))]),
            'le': torch.stack([train_input['target_le'][j].to(device) for j in range(len(train_input['target_le']))])
        }

        # compute output
        output = model(images, summaries, triples)

        # compute loss
        loss = criterion(output, target)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        if args.task == 'mtl': # measure performance and record loss
            losses['mtl'].update(loss['mtl'].data, args.batch_size)
            losses['ir'].update(loss['ir'].data, args.batch_size)
            losses['le'].update(loss['le'].data, args.batch_size)
            log_loss = f'IR Loss: {losses["ir"].val:.4f} ({losses["ir"].avg:.4f}) - LE Loss: {losses["le"].val:.4f} ({losses["le"].avg:.4f})'
            loss[args.task].backward()
        else:
            losses[args.task].update(loss.data, args.batch_size)
            log_loss = f'Loss: {losses[args.task].val:.4f} ({losses[args.task].avg:.4f})'
            loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        logger.info(f'Epoch: {epoch+1} - {log_loss} - Batch: {((i+1)/len(train_loader))*100:.2f}% - Time: {batch_time.sum:0.2f}s')

def validate(val_loader, model, criterion):
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
        'loss': losses[args.task].avg,
        'log': {
            'Loss': log_loss
        }
    }

    if args.task in ['ir', 'mtl']:
        medR, recall = rank(data0, data1, data2)
        results['log']['medR'] = medR
        results['log']['Recall'] = ' - '.join([f'{k}: {v}' for k, v in recall.items()])

    if args.task in ['le', 'mtl']:
        le_img = [p for pred in le_img for p in pred]
        le_txt = [p for pred in le_txt for p in pred]

        img_top1 = sum([1 if p[0] in p[1] else 0 for p in le_img])/len(le_img)
        img_top5 = sum([1 if p[0] in p[2] else 0 for p in le_img])/len(le_img)
        img_top10 = sum([1 if p[0] in p[3] else 0 for p in le_img])/len(le_img)

        txt_top1 = sum([1 if p[0] in p[1] else 0 for p in le_txt])/len(le_txt)
        txt_top5 = sum([1 if p[0] in p[2] else 0 for p in le_txt])/len(le_txt)
        txt_top10 = sum([1 if p[0] in p[3] else 0 for p in le_txt])/len(le_txt)

        results['log']['LE Image'] = f'Top@1: {img_top1:.2f} - Top@5: {img_top5:.2f} - Top@10: {img_top10:.2f}'
        results['log']['LE Text'] = f'Top@1: {txt_top1:.2f} - Top@5: {txt_top5:.2f} - Top@10: {txt_top10:.2f}'

    return results

if __name__ == '__main__':
    main()