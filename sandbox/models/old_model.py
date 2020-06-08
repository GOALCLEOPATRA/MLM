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

# coord network
class CoordNet(nn.Module):
    def __init__(self, tags=args.cell_dim, dropout=args.dropout):
        super(CoordNet, self).__init__()
        self.coord_net = nn.Sequential(
            nn.Linear(args.emb_dim, args.emb_dim),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(args.emb_dim, tags)
        )

    def forward(self, x):
        return self.coord_net(x.unsqueeze(1))

# embed images
class LearnImages(nn.Module):
    def __init__(self):
        super(LearnImages, self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels=args.img_dim, out_channels=args.emb_dim, kernel_size=1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(args.emb_dim, args.emb_dim),
            nn.Tanh(),
            Norm()
        )

    def forward(self, x):
        return self.embedding(x.unsqueeze(2))

# embed summaries
class LearnSummaries(nn.Module):
    def __init__(self):
        super(LearnSummaries, self).__init__()
        self.embedding = nn.Sequential(
            nn.LSTM(input_size=args.smr_dim, hidden_size=args.emb_dim, bidirectional=False, batch_first=True),
            LstmFlatten(),
            nn.ReLU(),
            nn.Linear(args.emb_dim, args.emb_dim),
            nn.Tanh(),
            Norm()
        )

    def forward(self, x):
        return self.embedding(x.unsqueeze(1))

# embed triples
class LearnTriples(nn.Module):
    def __init__(self, dropout=args.dropout):
        super(LearnTriples, self).__init__()
        self.embedding = nn.Sequential(
            nn.LSTM(input_size=args.smr_dim, hidden_size=args.emb_dim, bidirectional=False, batch_first=True),
            LstmFlatten(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(args.emb_dim, args.emb_dim),
            Norm()
        )

    def forward(self, x):
        return self.embedding(x.unsqueeze(1))

# MLM Baseline model
class MLMBaseline(nn.Module):
    def __init__(self):
        super(MLMBaseline, self).__init__()
        self.learn_img  = LearnImages()
        self.learn_sum  = LearnSummaries()
        self.learn_tri  = LearnTriples()
        self.coord_net  = CoordNet()
        self.fc1        = torch.nn.Linear(args.emb_dim + args.emb_dim, args.emb_dim)
        self.fc2        = torch.nn.Linear(args.cell_dim + args.cell_dim, args.cell_dim)

    def forward(self, image, summary, triple):
        # input embeddings
        img_emb = self.learn_img(image)
        sum_emb = self.learn_sum(summary)
        tri_emb = self.learn_tri(triple)

        # coord embedding
        img_coord = self.coord_net(img_emb)
        txt_coord = self.coord_net(sum_emb)
        tri_coord = self.coord_net(tri_emb)

        # combine text and triple
        # task IR
        txt_triple = torch.cat((sum_emb, tri_emb), 1)
        txt_triple = self.fc1(txt_triple)

        # task LE
        txt_coord_triple = torch.cat((txt_coord, tri_coord), 1)
        txt_coord_triple = self.fc2(txt_coord_triple)

        return {
            'ir': [img_emb, txt_triple],
            'le': [img_coord, txt_coord_triple]
        }
