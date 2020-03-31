import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='mlm')
    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no-cuda', action='store_true')

    # data
    parser.add_argument('--data_path', default='dataset/bert/')
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--input', default='image', type=str) # [image|coord] or [text|coord] query type

    # model
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--snapshots', default='experiments/snapshots',type=str)

    # MLM model
    parser.add_argument('--embDim', default=1024, type=int)
    parser.add_argument('--imgDim', default=2048, type=int)
    parser.add_argument('--coordDim', default=436, type=int)
    parser.add_argument('--sumDim', default=3072, type=int)
    parser.add_argument('--classDim', default=3072, type=int)
    parser.add_argument('--multi_loss', default=True,type=bool)

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
    parser.add_argument('--model_path', default='experiments/snapshots/model_e16_v-14.5000.pth.tar', type=str)

    # MedR / Recall@1 / Recall@5 / Recall@10
    parser.add_argument('--medr', default=10000, type=int)

    return parser
