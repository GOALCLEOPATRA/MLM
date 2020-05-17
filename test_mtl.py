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
from models.model_mtl import MLMRetrieval, L2Joint
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
    model_mtl.to(device)

    # define loss function (criterion) and optimizer
    # cosine similarity between embeddings -> input1, input2, target
    criterion_t1 = nn.CosineEmbeddingLoss(margin=0.1).to(device)
    # bce for classification
    criterion_t2 = nn.CrossEntropyLoss()

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
    kff(test_loader, model, criterion_t1, criterion_t2)

def kff(test_loader, model, criterion_t1, criterion_t2):
# def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    t1_losses = AverageMeter()
    t2_losses = AverageMeter()
    pred_t2_img = []
    pred_t2_txt = []

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
        cluster_target = torch.stack([test_input['cluster'][j].to(device) for j in range(len(test_input['cluster']))])

        cluster_target_idx = cluster_target.long().argmax(1)

        # ids
        ids = torch.stack([test_input['id'][j].to(device) for j in range(len(test_input['id']))])

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
        
        pred_t2_img.append([[t.argmax(), torch.topk(o, k=1)[1], torch.topk(o, k=5)[1], torch.topk(o, k=10)[1]] for o, t in zip(output[1], cluster_target)])
        pred_t2_txt.append([[t.argmax(), torch.topk(o, k=1)[1], torch.topk(o, k=5)[1], torch.topk(o, k=10)[1]] for o, t in zip(output[3], cluster_target)])
    pred_t2_img = [p for pred in pred_t2_img for p in pred]
    pred_t2_txt = [p for pred in pred_t2_txt for p in pred]

    medR, recall = rank(args, data0, data1, data2)

    print(f'* Task 1 loss {t1_losses.avg:.4f}')
    print(f'** Task 1 medR {medR:.4f} ----- Recall {recall}')

    print(f'Task 2 Loss: {"%.4f" % t2_losses.avg}')
    print(f'Image Predictions')
    print(f'==>Pred@1: {sum([1 if p[0] in p[1] else 0 for p in pred_t2_img])/len(pred_t2_img):.2f}')
    print(f'==>Pred@5: {sum([1 if p[0] in p[2] else 0 for p in pred_t2_img])/len(pred_t2_img):.2f}')
    print(f'==>Pred@10: {sum([1 if p[0] in p[3] else 0 for p in pred_t2_img])/len(pred_t2_img):.2f}')
    print(f'Text-Triple Predictions')
    print(f'==>Pred@1: {sum([1 if p[0] in p[1] else 0 for p in pred_t2_txt])/len(pred_t2_txt):.2f}')
    print(f'==>Pred@5: {sum([1 if p[0] in p[2] else 0 for p in pred_t2_txt])/len(pred_t2_txt):.2f}')
    print(f'==>Pred@10: {sum([1 if p[0] in p[3] else 0 for p in pred_t2_txt])/len(pred_t2_txt):.2f}')

    with open(f'{ROOT_PATH}/{args.path_results}/img_embeds.pkl', 'wb') as f:
        pickle.dump(data0, f)
    with open(f'{ROOT_PATH}/{args.path_results}/text_embeds.pkl', 'wb') as f:
        pickle.dump(data1, f)
    with open(f'{ROOT_PATH}/{args.path_results}/ids.pkl', 'wb') as f:
        pickle.dump(data2, f)

if __name__ == '__main__':
    model_mtl = L2Joint()
    main()
