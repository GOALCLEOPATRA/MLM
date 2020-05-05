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

# embed images
class LearnImages(nn.Module):
    def __init__(self):
        super(LearnImages, self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels=args.imgDim, out_channels=args.embDim, kernel_size=1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(args.embDim, args.embDim),
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
            nn.LSTM(input_size=args.wikiDim, hidden_size=args.embDim, bidirectional=False, batch_first=True),
            LstmFlatten(),
            nn.ReLU(),
            nn.Linear(args.embDim, args.embDim),
            nn.Tanh(),
            Norm()
        )

    def forward(self, x):
        return self.embedding(x.unsqueeze(1))


# MLM model
class MLMRetrieval(nn.Module):
    def __init__(self):
        super(MLMRetrieval, self).__init__()
        self.learn_img      = LearnImages()
        self.learn_sum      = LearnSummaries()

    def forward(self, input, sum):
        # input embedding
        input_emb = self.learn_img(input)

        # coord embedding
        sum_emb = self.learn_sum(sum)

        return [input_emb, sum_emb]
