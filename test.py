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
from models.model import MLMRetrieval

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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    print('Test loader prepared.')

    # run test
    test(test_loader, model, criterion)

def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    cos_losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, test_input in enumerate(test_loader):
        # inputs
        input_img = torch.stack([test_input['image'][j].to(device) for j in range(len(test_input['image']))])
        input_enwiki = torch.stack([test_input['en_wiki'][j].to(device) for j in range(len(test_input['en_wiki']))])
        input_frwiki = torch.stack([test_input['fr_wiki'][j].to(device) for j in range(len(test_input['fr_wiki']))])
        input_dewiki = torch.stack([test_input['de_wiki'][j].to(device) for j in range(len(test_input['de_wiki']))])
        input_coord = torch.stack([test_input['coord'][j].to(device) for j in range(len(test_input['coord']))])
        input_triples = torch.stack([test_input['triple'][j].to(device) for j in range(len(test_input['triple']))])

        # target
        target_var = torch.stack([test_input['target'][j].to(device) for j in range(len(test_input['target']))])

        # ids
        ids = torch.stack([test_input['id'][j].to(device) for j in range(len(test_input['id']))])

        # compute output
        output = model(input_img, input_enwiki, input_frwiki, input_dewiki, input_coord, input_triples)

        # compute loss
        loss = criterion(output[0], output[1], target_var)
        # measure performance and record loss
        cos_losses.update(loss.data, input_img.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i==0:
            data0 = output[0].data.cpu().numpy()
            data1 = output[1].data.cpu().numpy()
            data2 = ids.data.cpu().numpy()
        else:
            data0 = np.concatenate((data0, output[0].data.cpu().numpy()), axis=0)
            data1 = np.concatenate((data1, output[1].data.cpu().numpy()), axis=0)
            data2 = np.concatenate((data2, ids.data.cpu().numpy()), axis=0)

    medR, recall = rank(args, data0, data1, data2)

    print(f'* Test loss {cos_losses.avg:.4f}')
    print(f'** Test medR {medR:.4f} ----- Recall {recall}')

    with open(f'{ROOT_PATH}/{args.path_results}/img_embeds.pkl', 'wb') as f:
        pickle.dump(data0, f)
    with open(f'{ROOT_PATH}/{args.path_results}/text_embeds.pkl', 'wb') as f:
        pickle.dump(data1, f)
    with open(f'{ROOT_PATH}/{args.path_results}/ids.pkl', 'wb') as f:
        pickle.dump(data2, f)

if __name__ == '__main__':
    main()