import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='MLM')
    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no-cuda', action='store_true')

    # data
    parser.add_argument('--data_path', default='data/MLM_v1_sample')
    parser.add_argument('--workers', default=0, type=int)

    # model
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--snapshots', default='experiments/snapshots',type=str)

    # task
    parser.add_argument('--task', default='mtl', choices=['mtl', 'ir', 'le'], type=str)

    # MLM model
    parser.add_argument('--emb_dim', default=1024, type=int)
    parser.add_argument('--img_dim', default=4096, type=int)
    parser.add_argument('--cell_dim', default=116, type=int)
    parser.add_argument('--smr_dim', default=3072, type=int)
    parser.add_argument('--tpl_dim', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)

    # training
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--valfreq', default=1, type=int)
    parser.add_argument('--resume', default='', type=str)

    # test
    parser.add_argument('--path_results', default='experiments/results', type=str)
    parser.add_argument('--model_path', default='experiments/snapshots/MLM_v1_sample/mtl/epoch_49_loss_1.94.pth.tar', type=str)

    # MedR / Recall@1 / Recall@5 / Recall@10
    parser.add_argument('--emb_type', default='image', type=str) # [image|text] query type
    parser.add_argument('--medr', default=500, type=int)

    return parser
