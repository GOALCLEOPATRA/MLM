import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='mlm')
    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no-cuda', action='store_true')

    # data
    parser.add_argument('--data_path', default='data/')
    parser.add_argument('--workers', default=0, type=int)

    # model
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--snapshots', default='experiments/snapshots',type=str)

    # MLM model
    parser.add_argument('--embDim', default=1024, type=int)
    parser.add_argument('--imgDim', default=2048, type=int)
    parser.add_argument('--coordDim', default=2048, type=int)
    parser.add_argument('--wikiDim', default=2048, type=int)
    parser.add_argument('--tripleDim', default=2048, type=int)
    parser.add_argument('--preModel', default='resNet50',type=str)
    # will be used later
    # parser.add_argument('--max_triples', default=20, type=int)
    # parser.add_argument('--max_imgs', default=5, type=int)
    # parser.add_argument('--maxSeqlen', default=20, type=int)
    # parser.add_argument('--multi_loss', default=True,type=bool)

    # training
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--valfreq', default=10,type=int)
    parser.add_argument('--resume', default='', type=str)
    # might be used later
    # parser.add_argument('--patience', default=1, type=int)
    # parser.add_argument('--freeVision', default=False, type=bool)
    # parser.add_argument('--freeText', default=True, type=bool)

    # test
    parser.add_argument('--path_results', default='experiments/results', type=str)
    parser.add_argument('--model_path', default='experiments/snapshots/model_e16_v-14.5000.pth.tar', type=str)

    # MedR / Recall@1 / Recall@5 / Recall@10
    parser.add_argument('--embtype', default='image', type=str) # [image|text] query type
    parser.add_argument('--medr', default=30, type=int)

    return parser
