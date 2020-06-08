import os
import json
import random
import torch
import logging
import numpy as np
import urllib.parse
import torch.nn as nn
from pathlib import Path
from args import get_parser
from models.model import MLMBaseline
from data.data_loader import MLMLoader
from utils import IRLoss, LELoss, MTLLoss, AverageMeter

ROOT_PATH = Path(os.path.dirname(__file__))

# read parser
parser = get_parser()
args = parser.parse_args()

# set logger
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# set a seed value
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)

def main():
    # set model
    model = MLMBaseline()
    model.to(device)

    model_path = f'{ROOT_PATH}/{args.snapshots}/{args.data_path.split("/")[-1]}/{args.task}/{args.model_name}'
    logger.info(f"=> loading checkpoint '{model_path}'")
    if device.type == 'cpu':
        checkpoint = torch.load(model_path, encoding='latin1', map_location='cpu')
    else:
        checkpoint = torch.load(model_path, encoding='latin1')
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f"=> loaded checkpoint '{model_path}' (epoch {checkpoint['epoch']})")

    # prepare test loader
    test_loader = torch.utils.data.DataLoader(
        MLMLoader(data_path=f'{ROOT_PATH}/{args.data_path}', partition='test'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    logger.info('Test loader prepared.')

    # run test
    test(test_loader, model)

def test(val_loader, model):
    save_results = {}
    le_img = []
    le_txt = []

    # load raw data
    id_classes = {}
    id_labels = {}
    for i in range(1, 21):
        data_path = f'{str(ROOT_PATH.parent)}/mlm_dataset/MLM_v1/MLM_{i}/MLM_{i}.json'
        with open(data_path) as json_file:
            for d in json.load(json_file):
                id_classes[d['id']] = d['classes'][0]
                id_labels[d['id']] = urllib.parse.unquote(d['label'])

    # create cell coordinates
    h5f = MLMLoader(data_path=f'{ROOT_PATH}/{args.data_path}', partition='test').h5f
    cell_coords = {}
    for id in h5f['ids']:
        cell_id = np.argmax(h5f[f'{id}_onehot'][()]).tolist()
        coord = ','.join([str(fl) for fl in h5f[f'{id}_coordinates'][()].tolist()])
        cell_coords[cell_id] = coord

    # switch to evaluate mode
    model.eval()

    for i, val_input in enumerate(val_loader):
        # inputs
        images = torch.stack([val_input['image'][j].to(device) for j in range(len(val_input['image']))])
        summaries = torch.stack([val_input['summary'][j].to(device) for j in range(len(val_input['summary']))])
        classes = torch.stack([val_input['classes'][j].to(device) for j in range(len(val_input['classes']))])

        # target
        target = {
            'ir': torch.stack([val_input['target_ir'][j].to(device) for j in range(len(val_input['target_ir']))]),
            'le': torch.stack([val_input['target_le'][j].to(device) for j in range(len(val_input['target_le']))]),
            'ids': torch.stack([val_input['id'][j].to(device) for j in range(len(val_input['id']))])
        }

        # compute output
        output = model(images, summaries, classes)

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
            # images
            for o, t, id in zip(output['le'][0], target['le'], target['ids']):
                label = id_labels[id.tolist()]
                clss = id_classes[id.tolist()]
                trg = [t.tolist(), f'https://www.google.com/maps/place/{cell_coords[t.tolist()]}']
                rnk = torch.topk(o, k=args.cell_dim)[1].tolist().index(t.tolist())+1
                top10 = []
                for top in torch.topk(o, k=10)[1].tolist():
                    if top in cell_coords:
                        top10.append([top, f'https://www.google.com/maps/place/{cell_coords[top]}'])
                    else:
                        top10 = []
                        break
                if len(top10) == 10:
                    le_img.append({'id': id.tolist(), 'label': label, 'class': clss, 'target': trg, 'rank': rnk, 'top10': top10})

            # summaries
            for o, t, id in zip(output['le'][1], target['le'], target['ids']):
                label = id_labels[id.tolist()]
                clss = id_classes[id.tolist()]
                target = [t.tolist(), f'https://www.google.com/maps/place/{cell_coords[t.tolist()]}']
                rnk = torch.topk(o, k=args.cell_dim)[1].tolist().index(t.tolist())+1
                top10 = []
                for top in torch.topk(o, k=10)[1].tolist():
                    if top in cell_coords:
                        top10.append([top, f'https://www.google.com/maps/place/{cell_coords[top]}'])
                    else:
                        top10 = []
                        break
                if len(top10) == 10:
                    le_txt.append({'id': id.tolist(), 'label': label, 'class': clss, 'target': target, 'rank': rnk, 'top10': top10})

    if args.task in ['ir', 'mtl']:
        ir = ir_rank(data0, data1, data2, id_classes, id_labels)
        with open(f'{args.data_path.split("/")[-1]}_{args.task}_ir_{args.emb_type}.json', 'w', encoding='utf-8') as f:
            json.dump(ir, f, ensure_ascii=False, indent=4)

    if args.task in ['le', 'mtl']:
        with open(f'{args.data_path.split("/")[-1]}_{args.task}_le_image.json', 'w', encoding='utf-8') as f:
            json.dump(le_img[:2000], f, ensure_ascii=False, indent=4)

        with open(f'{args.data_path.split("/")[-1]}_{args.task}_le_summary.json', 'w', encoding='utf-8') as f:
            json.dump(le_txt[:2000], f, ensure_ascii=False, indent=4)

# ranking method for evaluating results
def ir_rank(img_embeds, text_embeds, names, id_classes, id_labels):
    # Sort based on names to always pick same samples for medr
    idxs = np.argsort(names)
    names = names[idxs]

    # Ranker
    N = args.medr
    idxs = range(N)

    ids = random.sample(range(0,len(names)), N)
    im_sub = img_embeds[ids,:]
    instr_sub = text_embeds[ids,:]
    ids_sub = names[ids]

    if args.emb_type == 'image':
        sims = np.dot(im_sub, instr_sub.T) # for im2text
    else:
        sims = np.dot(instr_sub, im_sub.T) # for text2im

    ranked_results = []

    for ii in idxs:
        # get a column of similarities
        sim = sims[ii, :]

        # sort indices in descending order
        sorting = np.argsort(sim)[::-1].tolist()

        # find where the index of the pair sample ended up in the sorting
        pos = sorting.index(ii)

        name = ids_sub[ii].item()
        top10_ids = ids_sub[sorting[:10]].tolist()

        ranked_results.append({
            'id': name,
            'label': id_labels[name],
            'class': id_classes[name],
            'target': f'https://www.wikidata.org/wiki/Q{name}',
            'rank': pos+1,
            'top10': [f'https://www.wikidata.org/wiki/Q{id}' for id in top10_ids],
        })

    return ranked_results

if __name__ == '__main__':
    main()