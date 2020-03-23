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
from utils import AverageMeter, rank, save_checkpoint, adjust_learning_rate

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
    # cosine similarity between embeddings -> input1, input2, target
    criterion = nn.CosineEmbeddingLoss(margin=0.1).to(device)

    # we can set two groups of parameters(one for each modality) and update one of those during training.
    # the idea is after some epochs if there is no improvement on validation set
    # then we switch and update the other parameters group. On group can be updated at a time.

    # creating different parameter groups
    # vision_params = list(map(id, model.img2vec.visionMLP.parameters()))
    # base_params   = filter(lambda p: id(p) not in vision_params, model.parameters())

    # optimizer - with lr initialized accordingly
    # optimizer = torch.optim.Adam([
    #             {'params': base_params},
    #             {'params': model.img2vec.visionMLP.parameters(), 'lr': args.lr * args.freeVision }
    #         ], lr=args.lr * args.freeText)

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
    # print(f'There are {optimizer.param_groups} parameter groups')
    # print(f'Initial base params lr: {optimizer.param_groups[0]['lr']})
    # print(f'Initial vision params lr: {optimizer.param_groups[1]['lr']}')

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
        if (epoch+1) % args.valfreq == 0 and epoch != 0:
            val_loss = validate(val_loader, model, criterion)

            # check patience
            # if val_loss >= best_val:
            #     valtrack += 1
            # else:
            #     valtrack = 0
            # if valtrack >= args.patience:
            #     # we switch modalities
            #     args.freeVision = args.freeText; args.freeText = not(args.freeVision)
            #     # change the learning rate accordingly
            #     adjust_learning_rate(optimizer, epoch, args)
            #     valtrack = 0

            # save the best model
            if val_loss < best_val:
              best_val = min(val_loss, best_val)
              save_checkpoint({
                  'epoch': epoch + 1,
                  'state_dict': model.state_dict(),
                  'best_val': best_val,
                  'optimizer': optimizer.state_dict(),
                  'curr_val': val_loss})

            print(f'** Validation: {best_val} (best)')

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cos_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, train_input in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # inputs
        input_img = torch.stack([train_input['image'][j].to(device) for j in range(len(train_input['image']))])
        input_enwiki = torch.stack([train_input['en_wiki'][j].to(device) for j in range(len(train_input['en_wiki']))])
        input_frwiki = torch.stack([train_input['fr_wiki'][j].to(device) for j in range(len(train_input['fr_wiki']))])
        input_dewiki = torch.stack([train_input['de_wiki'][j].to(device) for j in range(len(train_input['de_wiki']))])
        input_coord = torch.stack([train_input['coord'][j].to(device) for j in range(len(train_input['coord']))])
        input_triples = torch.stack([train_input['triple'][j].to(device) for j in range(len(train_input['triple']))])

        # target
        target_var = torch.stack([train_input['target'][j].to(device) for j in range(len(train_input['target']))])

        # compute output
        output = model(input_img, input_enwiki, input_frwiki, input_dewiki, input_coord, input_triples)

        # compute loss
        loss = criterion(output[0], output[1], target_var)
        # measure performance and record loss
        cos_losses.update(loss.data, input_img.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('-----------------------------------------------------------------------')
        print(f'Epoch: {epoch+1} ----- Loss: {cos_losses.val:.4f} ({cos_losses.avg:.4f})')

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    cos_losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, val_input in enumerate(val_loader):
        # inputs
        input_img = torch.stack([val_input['image'][j].to(device) for j in range(len(val_input['image']))])
        input_enwiki = torch.stack([val_input['en_wiki'][j].to(device) for j in range(len(val_input['en_wiki']))])
        input_frwiki = torch.stack([val_input['fr_wiki'][j].to(device) for j in range(len(val_input['fr_wiki']))])
        input_dewiki = torch.stack([val_input['de_wiki'][j].to(device) for j in range(len(val_input['de_wiki']))])
        input_coord = torch.stack([val_input['coord'][j].to(device) for j in range(len(val_input['coord']))])
        input_triples = torch.stack([val_input['triple'][j].to(device) for j in range(len(val_input['triple']))])

        # ids
        ids = torch.stack([val_input['id'][j].to(device) for j in range(len(val_input['id']))])

        # compute output
        output = model(input_img, input_enwiki, input_frwiki, input_dewiki, input_coord, input_triples)

        if i==0:
            data0 = output[0].data.cpu().numpy()
            data1 = output[1].data.cpu().numpy()
            data2 = ids.data.cpu().numpy()
        else:
            data0 = np.concatenate((data0, output[0].data.cpu().numpy()), axis=0)
            data1 = np.concatenate((data1, output[1].data.cpu().numpy()), axis=0)
            data2 = np.concatenate((data2, ids.data.cpu().numpy()), axis=0)

    medR, recall = rank(args, data0, data1, data2)

    print(f'* Val medR {medR:.4f} ----- Recall {recall}')

    return medR

if __name__ == '__main__':
    main()