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
from models.model_t1 import MLMRetrieval
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

    # define loss function (criterion_t1) and optimizer
    # cosine similarity between embeddings -> input1, input2, target
    criterion_t1 = nn.CosineEmbeddingLoss(margin=0.1).to(device)

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
    kff(test_loader, model, criterion_t1)

def kff(test_loader, model, criterion_t1):
# def test(test_loader, model, criterion_t1):
    batch_time = AverageMeter()
    t1_losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, test_input in enumerate(test_loader):
                # inputs

        input_img = torch.stack([test_input['image'][j].to(device) for j in range(len(test_input['image']))])
        input_summary = torch.stack([test_input['multi_wiki'][j].to(device) for j in range(len(test_input['multi_wiki']))])
        input_tri =  torch.stack([test_input['triple'][j].to(device) for j in range(len(test_input['triple']))])

        # target
        target_var = torch.stack([test_input['target'][j].to(device) for j in range(len(test_input['target']))])

        # ids
        ids = torch.stack([test_input['id'][j].to(device) for j in range(len(test_input['id']))])

        # compute output

        output = model(input_img, input_summary, input_tri)
        # compute loss
        m_loss_t1 = criterion_t1(output[0], output[2], target_var)
        # measure performance and record loss
        t1_losses.update(m_loss_t1.data, input_img.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i==0:
            data0 = output[0].data.cpu().numpy()
            data1 = output[2].data.cpu().numpy()
            data2 = ids.data.cpu().numpy()
        else:
            data0 = np.concatenate((data0, output[0].data.cpu().numpy()), axis=0)
            data1 = np.concatenate((data1, output[2].data.cpu().numpy()), axis=0)
            data2 = np.concatenate((data2, ids.data.cpu().numpy()), axis=0)

    medR, recall = rank(args, data0, data1, data2)

    print(f'* Test loss {t1_losses.avg:.4f}')
    print(f'** Test medR {medR:.4f} ----- Recall {recall}')

    with open(f'{ROOT_PATH}/{args.path_results}/img_embeds.pkl', 'wb') as f:
        pickle.dump(data0, f)
    with open(f'{ROOT_PATH}/{args.path_results}/text_embeds.pkl', 'wb') as f:
        pickle.dump(data1, f)
    with open(f'{ROOT_PATH}/{args.path_results}/ids.pkl', 'wb') as f:
        pickle.dump(data2, f)

if __name__ == '__main__':
    main()