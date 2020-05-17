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
from models.model_mtl import MLMRetrieval, L2Joint
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
    model_mtl.to(device)

    # define loss functions (criterion) and optimizer
    # cosine similarity between embeddings -> input1, input2, target
    criterion_t1 = nn.CosineEmbeddingLoss(margin=0.1).to(device)
    # bce for classification
    criterion_t2 = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}''")
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        t1_v = checkpoint('t1_v')
        t2_v = checkpoint('t2_v')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        print(f"=> no checkpoint found at '{args.resume}'")
    else:
        t1_v = float('inf')
        t2_v = float('inf')

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
        train(train_loader, model, criterion_t1, criterion_t2, optimizer, epoch)

        # evaluate on validation set
        if (epoch+1) % args.valfreq == 0:
            medR, recall, m_loss_t1, m_loss_t2 = validate(val_loader, model, criterion_t1, criterion_t2, epoch)
            t1_v = medR
            t2_v = m_loss_t2

            # save all models
            save_checkpoint({
                'task_id': task_id,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                't1_v': t1_v,
                't2_v': t2_v})

            print(f'**Val -- medR T1: {medR} -- Recall T1 {recall} -- Loss T1: {"%.4f" % m_loss_t1} -- Loss T2: {"%.4f" % m_loss_t2}')

            # visualise
            m_loss_t1_np = m_loss_t1.cpu().numpy()
            plotter.plot('loss t1', 'val t1', 'Loss for T1', epoch + 1, m_loss_t1_np)
            medR_loss_np = medR
            plotter.plot('medR t1', 'val medR', 'Val medR for T1', epoch + 1, medR_loss_np)
            m_loss_t2_np = m_loss_t2.cpu().numpy()
            plotter.plot('loss t2', 'val t2', 'Loss for T2', epoch + 1, m_loss_t2_np)



def train(train_loader, model, criterion_t1, criterion_t2, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    t1_losses = AverageMeter()
    t2_losses = AverageMeter()
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
        cluster_target = torch.stack([train_input['cluster'][j].to(device) for j in range(len(train_input['cluster']))])

        cluster_target_idx = cluster_target.long().argmax(1)

        # compute output
        output = model(input_img, input_summary, input_tri)
        
        t1_a_i = output[0]
        t1_c_i = output[2]
        t2_a_i = output[1]
        t2_c_i = output[3]

        # compute loss 
        mtl = model_mtl(t1_a_i, t2_a_i, target_var, cluster_target_idx, t1_c_i, t2_c_i)
        m_loss_t1 = mtl[1]
        m_loss_t2 = mtl[2]
        wa_losses_mn = mtl[0]
        
        # measure performance and record loss
        t1_losses.update(m_loss_t1.data, input_img.size(0))
        t2_losses.update(m_loss_t2.data, input_img.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        wa_losses_mn.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('-----------------------------------------------------------------------')
        print(f'Epoch: {epoch+1} ----- T1 Loss: {t1_losses.val:.4f} ({t1_losses.avg:.4f}) - T2 Loss: {t2_losses.val:.4f} ({t2_losses.avg:.4f}) - Batch done: {((i+1)/len(train_loader))*100:.2f}% - Time: {batch_time.sum:0.2f}s')
    
    # Visualise
    t1_losses_np_avg = t1_losses.avg.cpu().numpy()
    plotter.plot('loss t1', 'train t1', 'Loss for T1', epoch + 1, t1_losses_np_avg)
    t2_losses_np_avg = t2_losses.avg.cpu().numpy()
    plotter.plot('loss t2', 'train t2', 'Loss for T2', epoch + 1, t2_losses_np_avg)


def validate(val_loader, model, criterion_t1, criterion_t2, epoch):
    batch_time = AverageMeter()
    t1_losses = AverageMeter()
    t2_losses = AverageMeter()

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
        target_var = torch.stack([val_input['target'][j].to(device) for j in range(len(val_input['target']))])
        cluster_target = torch.stack([val_input['cluster'][j].to(device) for j in range(len(val_input['cluster']))])

        cluster_target_idx = cluster_target.long().argmax(1)

        # compute output
        output = model(input_img, input_summary, input_tri)

        t1_a_i = output[0]
        t1_c_i = output[2]
        t2_a_i = output[1]
        t2_c_i = output[3]

        # compute losses for mtl
        mtl = model_mtl(t1_a_i, t2_a_i, target_var, cluster_target_idx, t1_c_i, t2_c_i)
        m_loss_t1 = mtl[1]
        m_loss_t2 = mtl[2]
        wa_losses_mn = mtl[0]

        t1_losses.update(m_loss_t1.data, cluster_target.size(0))
        t2_losses.update(m_loss_t2.data, cluster_target.size(0))

        # losses
        # task 1
        if i==0:
            data0 = output[0].data.cpu().numpy()
            data1 = output[2].data.cpu().numpy()
            data2 = ids.data.cpu().numpy()
        else:
            data0 = np.concatenate((data0, output[0].data.cpu().numpy()), axis=0)
            data1 = np.concatenate((data1, output[2].data.cpu().numpy()), axis=0)
            data2 = np.concatenate((data2, ids.data.cpu().numpy()), axis=0)

    medR, recall = rank(args, data0, data1, data2)

    return medR, recall, t1_losses.avg, t2_losses.avg

if __name__ == '__main__':
    # identifier
    task_id = 'mtl'
    # visualise
    plotter = VisdomLinePlotter()
    # instantiate mtl model
    model_mtl = L2Joint()
    main()