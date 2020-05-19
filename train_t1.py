import os
import time
import random
import torch
import torch.optim
import torch.nn as nn
import numpy as np
from pathlib import Path
from visdom import Visdom
from args import get_parser
from models.model_t1 import MLMRetrieval
from data.data_loader_coord_cluster import MLMLoader
from utils import AverageMeter, rank, save_checkpoint, VisdomLinePlotter

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

    # define loss function (criterion_t1) and optimizer
    # cosine similarity between embeddings -> input1, input2, target
    criterion_t1 = nn.CosineEmbeddingLoss(margin=0.1).to(device)

    # optimizer - one group parameters for now
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}''")
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_val = checkpoint['best_val']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
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
        batch_size=1000,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    print('Validation loader prepared.')

    valtrack = 0

    # run epochs
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model, criterion_t1, optimizer, epoch)

        # evaluate on validation set
        if (epoch+1) % args.valfreq == 0:
            medR, recall = validate(val_loader, model, criterion_t1, epoch)
            val_loss = medR
            if val_loss < best_val:
              best_val = min(val_loss, best_val)


            # save the best model
            save_checkpoint({
                'task_id': task_id,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val': best_val,
                'medR': medR,
                'optimizer': optimizer.state_dict(),
                'curr_val': val_loss})

            print(f'**Val -- medR T1: {"%.4f" % medR} -- recall T1: {recall} -- medR T1 (best): {"%.4f" % best_val} (best)')

            # visualise
            medR_loss_np = medR
            plotter.plot('medR t1', 'val medR', 'Val medR for T1', epoch + 1, medR_loss_np)


def train(train_loader, model, criterion_t1, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    t1_losses = AverageMeter()
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
        input_summary = torch.stack([train_input['multi_wiki'][j].to(device) for j in range(len(train_input['multi_wiki']))])
        input_tri =  torch.stack([train_input['triple'][j].to(device) for j in range(len(train_input['triple']))])

        # target
        target_var = torch.stack([train_input['target'][j].to(device) for j in range(len(train_input['target']))])

        #print('target_var', target_var)

        # compute output
        output = model(input_img, input_summary, input_tri)

        # compute loss
        m_loss_t1 = criterion_t1(output[0], output[2], target_var)
        # measure performance and record loss
        t1_losses.update(m_loss_t1.data, input_img.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        m_loss_t1.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('-----------------------------------------------------------------------')
        print(f'Epoch: {epoch+1} ----- Loss: {t1_losses.val:.4f} ({t1_losses.avg:.4f}) - Batch done: {((i+1)/len(train_loader))*100:.2f}% - Time: {batch_time.sum:0.2f}s')
    
    # Visualise
    t1_losses_np_avg = t1_losses.avg.cpu().numpy()
    plotter.plot('loss t1', 'train t1', 'Loss for T1', epoch + 1, t1_losses_np_avg)

def validate(val_loader, model, criterion_t1, epoch):
    batch_time = AverageMeter()
    t1_losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, val_input in enumerate(val_loader):
        # inputs
        input_img = torch.stack([val_input['image'][j].to(device) for j in range(len(val_input['image']))])
        input_summary = torch.stack( [val_input['multi_wiki'][j].to(device) for j in range(len(val_input['multi_wiki']))])
        input_tri =  torch.stack([val_input['triple'][j].to(device) for j in range(len(val_input['triple']))])

        # ids
        ids = torch.stack([val_input['id'][j].to(device) for j in range(len(val_input['id']))])

        # compute output
        output = model(input_img, input_summary, input_tri)

        if i==0:
            data0 = output[0].data.cpu().numpy()
            data1 = output[2].data.cpu().numpy()
            data2 = ids.data.cpu().numpy()
        else:
            data0 = np.concatenate((data0, output[0].data.cpu().numpy()), axis=0)
            data1 = np.concatenate((data1, output[2].data.cpu().numpy()), axis=0)
            data2 = np.concatenate((data2, ids.data.cpu().numpy()), axis=0)

    medR, recall = rank(args, data0, data1, data2)

    # write validation results on txt file
    f = open(f'experiments/results/img2text.txt', 'a')
    f.write(f'Epoch {epoch}: * Val medR {medR:.4f} ----- Recall {recall}\n')
    f.close()

    return medR, recall

if __name__ == '__main__':
    # identifier
    task_id = 't1'
    # visualise
    plotter = VisdomLinePlotter()
    main()