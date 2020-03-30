import torch
import torch.nn as nn
from args import get_parser

# read parser
parser = get_parser()
args = parser.parse_args()

def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

# embed images
class LearnImages(nn.Module):
    def __init__(self):
        super(LearnImages, self).__init__()
        self.visual_embedding = nn.Sequential(
            nn.Linear(args.imgDim, args.embDim),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.visual_embedding(x)
        x = norm(x)
        return x

# embed summaries
class LearnSummaries(nn.Module):
    def __init__(self):
        super(LearnSummaries, self).__init__()
        self.lstm = nn.LSTM(input_size=args.sumDim, hidden_size=args.embDim, bidirectional=False, batch_first=True)
        self.sum_embedding = nn.Sequential(
            nn.Linear(args.embDim, args.embDim),
            nn.Tanh(),
        )

    def forward(self, x):
        x, hidden = self.lstm(x.unsqueeze(1))
        x = self.sum_embedding(x)
        x = norm(x)
        return x.squeeze(1)

# embed coordinates
class LearnCoordinates(nn.Module):
    def __init__(self):
        super(LearnCoordinates, self).__init__()
        self.coord_embedding = nn.Sequential(
            nn.Linear(args.coordDim, args.embDim),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.coord_embedding(x)
        x = norm(x)
        return x

# embed classes
class LearnClasses(nn.Module):
    def __init__(self):
        super(LearnClasses, self).__init__()
        self.class_embedding = nn.Sequential(
            nn.Linear(args.classDim, args.embDim),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.class_embedding(x)
        x = norm(x)
        return x

# MLM model
class MLMRetrieval(nn.Module):
    def __init__(self):
        super(MLMRetrieval, self).__init__()
        self.learn_img      = LearnImages()
        self.learn_sum      = LearnSummaries()
        self.learn_coord    = LearnCoordinates()
        self.learn_cls      = LearnClasses()

    def forward(self, input, coord):
        # input embedding
        input_emb = self.learn_img(input) if args.input == 'image' else self.learn_sum(input)

        # coord embedding
        coord_emb = self.learn_coord(coord)

        # final output
        output = [input_emb, coord_emb]

        return output
