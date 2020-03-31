import os
import time
import random
import torch
import torch.optim
import torch.nn as nn
import numpy as np
from pathlib import Path
from args import get_parser
from models.model import MLMRetrieval
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
torch.cuda.set_device(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # load model
    model = MLMRetrieval()
    model.to(device)

    # define loss function (criterion) and optimizer
    # cosine similarity between embeddings -> input1, input2, target
    criterion = nn.CosineEmbeddingLoss(margin=0.1).to(device)

    # optimizer - one group parameters for now
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
            val_loss = validate(val_loader, model, criterion, epoch+1)
            # save the best model
            if val_loss < best_val:
              best_val = min(val_loss, best_val)
              save_checkpoint({
                  'epoch': epoch + 1,
                  'state_dict': model.state_dict(),
                  'best_val': best_val,
                  'optimizer': optimizer.state_dict(),
                  'curr_val': val_loss}, input_name=args.input)

            print(f'** Validation: {best_val} (best)')

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cos_losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, train_input in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # inputs
        input = torch.stack([train_input['image'][j].to(device) for j in range(len(train_input['image']))]) if args.input == 'image' else \
                torch.stack([train_input['multi_wiki'][j].to(device) for j in range(len(train_input['multi_wiki']))])
        input_coord = torch.stack([train_input['coord'][j].to(device) for j in range(len(train_input['coord']))])

        # target
        target_var = torch.stack([train_input['target'][j].to(device) for j in range(len(train_input['target']))])

        # compute output
        output = model(input, input_coord)

        # compute loss
        loss = criterion(output[0], output[1], target_var)
        # measure performance and record loss
        cos_losses.update(loss.data, input.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('-----------------------------------------------------------------')
        print(f'Epoch: {epoch+1} -- Loss: {cos_losses.val:.4f} ({cos_losses.avg:.4f}) -- Batch done: {((i+1)/len(train_loader))*100:.2f}% -- Time: {batch_time.sum:0.2f}s')

def validate(val_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, val_input in enumerate(val_loader):
        # inputs
        input = torch.stack([val_input['image'][j].to(device) for j in range(len(val_input['image']))]) if args.input == 'image' else \
                torch.stack([val_input['multi_wiki'][j].to(device) for j in range(len(val_input['multi_wiki']))])
        input_coord = torch.stack([val_input['coord'][j].to(device) for j in range(len(val_input['coord']))])

        # ids
        ids = torch.stack([val_input['id'][j].to(device) for j in range(len(val_input['id']))])

        # compute output
        output = model(input, input_coord)

        if i==0:
            data0 = output[0].data.cpu().numpy()
            data1 = output[-1].data.cpu().numpy()
            data2 = ids.data.cpu().numpy()
        else:
            data0 = np.concatenate((data0, output[0].data.cpu().numpy()), axis=0)
            data1 = np.concatenate((data1, output[-1].data.cpu().numpy()), axis=0)
            data2 = np.concatenate((data2, ids.data.cpu().numpy()), axis=0)

    medR, recall = rank(args, data0, data1, data2)

    print('-----------------------------------------------------------------')
    print(f'* Val medR {medR:.4f} ----- Recall {recall}')
    # write validation results on txt file
    f = open(f'experiments/results/{args.input}2coord.txt', 'a')
    f.write(f'Epoch {epoch}: * Val medR {medR:.4f} ----- Recall {recall}\n')
    f.close()

    return medR

if __name__ == '__main__':
    main()