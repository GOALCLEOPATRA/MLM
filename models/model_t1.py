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

# embed triples
class LearnTriples(nn.Module):
    def __init__(self, dropout=args.dropout):
        super(LearnTriples, self).__init__()
        self.embedding = nn.Sequential(
            nn.LSTM(input_size=args.wikiDim, hidden_size=args.embDim, bidirectional=False, batch_first=True),
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
class MLMRetrieval(nn.Module):
    def __init__(self):
        super(MLMRetrieval, self).__init__()
        self.learn_img      = LearnImages()
        self.learn_sum      = LearnSummaries()
        self.learn_tri      = LearnTriples()
        self.fc1 = torch.nn.Linear(args.embDim+args.embDim, args.embDim)
        

    def forward(self, input, sum, triple):
        # input embeddings
        input_emb = self.learn_img(input)
        sum_emb = self.learn_sum(sum)
        tri_emb = self.learn_tri(triple)
        
        # text embedding

        # combine text and triple
        sum_tri_t1 = torch.cat((sum_emb, tri_emb), 1)
        sum_tri_t1 = self.fc1(sum_tri_t1)

        return [input_emb, sum_emb, sum_tri_t1]