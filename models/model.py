import torch
import torch.nn as nn
from args import get_parser

# read parser
parser = get_parser()
args = parser.parse_args()

class Norm(nn.Module):
    def forward(self, input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

class LstmFlatten(nn.Module):
    def forward(self, x):
        return x[0].squeeze(1)

class CoordNet(nn.Module):
    def __init__(self, tags=args.coordDim, dropout=args.dropout):
        super(CoordNet, self).__init__()
        self.coord_net = nn.Sequential(
            nn.LSTM(input_size=args.embDim, hidden_size=args.embDim, bidirectional=False, batch_first=True),
            LstmFlatten(),
            nn.Dropout(dropout),
            nn.Linear(args.embDim, tags),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.coord_net(x.unsqueeze(1))

# embed images
class LearnImages(nn.Module):
    def __init__(self, dropout=args.dropout):
        super(LearnImages, self).__init__()
        self.embedding = nn.Sequential(
            nn.LSTM(input_size=args.imgDim, hidden_size=args.embDim, bidirectional=False, batch_first=True),
            LstmFlatten(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(args.embDim, args.embDim),
            Norm(),
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.embedding(x.unsqueeze(1))

# embed summaries
class LearnSummaries(nn.Module):
    def __init__(self, dropout=args.dropout):
        super(LearnSummaries, self).__init__()
        self.embedding = nn.Sequential(
            nn.LSTM(input_size=args.sumDim, hidden_size=args.embDim, bidirectional=False, batch_first=True),
            LstmFlatten(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(args.embDim, args.embDim),
            Norm(),
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.embedding(x.unsqueeze(1))

# MLM model
class MLMCoordPrediction(nn.Module):
    def __init__(self):
        super(MLMCoordPrediction, self).__init__()
        self.learn_img      = LearnImages()
        self.learn_sum      = LearnSummaries()
        self.image_coord    = CoordNet()
        self.text_coord     = CoordNet()

    def forward(self, image, text):
        # input embedding
        img_emb = self.learn_img(image)
        text_emb = self.learn_sum(text)

        # coord pred
        img_coord = self.image_coord(img_emb)
        txt_coord = self.text_coord(text_emb)

        return [img_coord, txt_coord]
