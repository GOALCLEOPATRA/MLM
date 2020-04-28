import os
import time
import random
import torch
import torch.optim
import torch.nn as nn
import numpy as np
from pathlib import Path
from args import get_parser
from models.model import MLMCoordPrediction
from data.data_loader import MLMLoader
from utils import AverageMeter, rank, save_checkpoint

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
torch.cuda.set_device(3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # load model
    model = MLMCoordPrediction()
    model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}''")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_val = checkpoint['best_val']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
            best_val = float('inf')
    else:
        best_val = float('inf')

    print(f'Initial model params lr: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # prepare training loader
    train_loader = torch.utils.data.DataLoader(
        MLMLoader(data_path=f'{ROOT_PATH}/{args.data_path}', partition='train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    print('Training loader prepared.')

    # prepared validation loader
    val_loader = torch.utils.data.DataLoader(
        MLMLoader(data_path=f'{ROOT_PATH}/{args.data_path}', partition='val'), # for now test, later we change to val
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    print('Validation loader prepared.')

    valtrack = 0

    # run epochs
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch+1) % args.valfreq == 0:
            img_loss, txt_loss = validate(val_loader, model, criterion, epoch+1)
            val_loss = img_loss + txt_loss
            # save the best model
            if val_loss < best_val:
              best_val = min(val_loss, best_val)
              save_checkpoint({
                  'epoch': epoch + 1,
                  'state_dict': model.state_dict(),
                  'best_val': best_val,
                  'optimizer': optimizer.state_dict(),
                  'curr_val': val_loss})

            print(f'** Validation: Image Loss: {img_loss}, Text Loss: {txt_loss} -- Total: {val_loss}, Best: {best_val}')

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    img_losses = AverageMeter()
    txt_losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, train_input in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # inputs
        image = torch.stack([train_input['image'][j].to(device) for j in range(len(train_input['image']))])
        text = torch.stack([train_input['multi_wiki'][j].to(device) for j in range(len(train_input['multi_wiki']))])

        # target
        coord_target = torch.stack([train_input['coord'][j].to(device) for j in range(len(train_input['coord']))])

        # compute output
        img_coord, txt_coord = model(image, text)

        # compute loss
        img_loss = criterion(img_coord, coord_target)
        txt_loss = criterion(txt_coord, coord_target)

        # measure performance and record loss
        img_losses.update(img_loss.data, image.size(0))
        txt_losses.update(txt_loss.data, text.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        img_loss.backward()
        txt_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('-----------------------------------------------------------------')
        print(f'Epoch: {epoch+1} - Image Loss: {img_losses.val:.4f} ({img_losses.avg:.4f}) - Text Loss: {txt_losses.val:.4f} ({txt_losses.avg:.4f}) - Batch done: {((i+1)/len(train_loader))*100:.2f}% - Time: {batch_time.sum:0.2f}s')

def validate(val_loader, model, criterion, epoch):
    img_losses = AverageMeter()
    txt_losses = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, val_input in enumerate(val_loader):
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

    return img_losses.avg, txt_losses.avg

if __name__ == '__main__':
    main()