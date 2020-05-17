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
from data.data_loader_coord_cluster import MLMLoader
from models.model_t2 import MLMRetrieval
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

    # define loss functions (criterion) and optimizer
    # bce for classification
    criterion_t2 = nn.CrossEntropyLoss()

    # optimizer
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
        MLMLoader(data_path=f'{ROOT_PATH}/{args.data_path}', partition='val'),
        batch_size=1000,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    print('Validation loader prepared.')

    valtrack = 0

    # run epochs
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model, criterion_t2, optimizer, epoch)

        # evaluate on validation set
        if (epoch+1) % args.valfreq == 0:
            m_loss_t2 = validate(val_loader, model, criterion_t2, epoch)
            val_loss = m_loss_t2
            if val_loss < best_val:
              best_val = min(val_loss, best_val)

            # save the best model
            save_checkpoint({
                'task_id': task_id,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val': best_val,
                'optimizer': optimizer.state_dict(),
                'curr_val': val_loss})

            print(f'** -- Loss T2: {"%.4f" % m_loss_t2} -- Loss T2 (best): {"%.4f" % best_val} (best)')

            # visualise
            m_loss_t2_np = m_loss_t2.cpu().numpy()
            plotter.plot('loss t2', 'val t2', 'Loss for T2', epoch + 1, m_loss_t2_np)



def train(train_loader, model, criterion_t2, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    t2_losses = AverageMeter()

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
        cluster_target = torch.stack([train_input['cluster'][j].to(device) for j in range(len(train_input['cluster']))])

        cluster_target_idx = cluster_target.long().argmax(1)

        # compute output
        output = model(input_img, input_summary, input_tri)

        # compute loss
        m_loss_t2_img = criterion_t2(output[0], cluster_target_idx)
        m_loss_t2_txt = criterion_t2(output[2], cluster_target_idx)
        m_loss_t2 = m_loss_t2_img + m_loss_t2_txt
        
        # measure performance and record loss
        t2_losses.update(m_loss_t2.data, cluster_target.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        m_loss_t2.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('-----------------------------------------------------------------------')
        print(f'Epoch: {epoch+1} -----  T2 Loss: {t2_losses.val:.4f} ({t2_losses.avg:.4f}) - Batch done: {((i+1)/len(train_loader))*100:.2f}% - Time: {batch_time.sum:0.2f}s')
    
    # Visualise
    t2_losses_np_avg = t2_losses.avg.cpu().numpy()
    plotter.plot('loss t2', 'train t2', 'Loss for T2', epoch + 1, t2_losses_np_avg)


def validate(val_loader, model, criterion_t2, epoch):
    batch_time = AverageMeter()
    t2_losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, val_input in enumerate(val_loader):
        # inputs
        input_img = torch.stack([val_input['image'][j].to(device) for j in range(len(val_input['image']))])
        input_summary = torch.stack( [val_input['multi_wiki'][j].to(device) for j in range(len(val_input['multi_wiki']))])
        input_tri =  torch.stack([val_input['triple'][j].to(device) for j in range(len(val_input['triple']))])

        # outputs
        target_var = torch.stack([val_input['target'][j].to(device) for j in range(len(val_input['target']))])
        cluster_target = torch.stack([val_input['cluster'][j].to(device) for j in range(len(val_input['cluster']))])

        cluster_target_idx = cluster_target.long().argmax(1)

        # compute output
        output = model(input_img, input_summary, input_tri)

        # compute losses
        m_loss_t2_img = criterion_t2(output[0], cluster_target_idx)
        m_loss_t2_txt = criterion_t2(output[2], cluster_target_idx)
        m_loss_t2 = m_loss_t2_img + m_loss_t2_txt

        # measure performance and record loss
        t2_losses.update(m_loss_t2.data, cluster_target.size(0))

    return t2_losses.avg

if __name__ == '__main__':
    # identifier
    task_id = 't2'
    # visualise
    plotter = VisdomLinePlotter()
    main()