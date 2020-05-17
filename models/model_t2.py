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
            nn.Linear(args.embDim, tags)
            #nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        return self.coord_net(x.unsqueeze(1))

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
            Norm()
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.embedding(x.unsqueeze(1))

# mtl model
class L2Joint(nn.Module):
    def __init__(self):
        super(L2Joint, self).__init__()
        self.mml_emp = torch.Tensor([True, False])
        self.n_mml = len(self.mml_emp)
        self.log_vars = torch.nn.Parameter(torch.zeros(self.n_mml))
        #self.mse = nn.MSELoss()
        self.t1_wa = torch.Tensor(1).requires_grad_(True)
        self.t2_wa = torch.Tensor(1).requires_grad_(True)

    def forward(self, m_loss_t1, m_loss_t2):
        t1 = m_loss_t1
        t2 = m_loss_t2

        # weighted average
        mm_losses = torch.stack((t1, t2))
        dtype = mm_losses.dtype
        device = mm_losses.device
        stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)
        weights = 1 / ((self.mml_emp.to(device).to(dtype)+1)*(stds**2))
        wa_losses = weights*mm_losses + torch.log(stds)
        wa_losses_mn = wa_losses.mean()
        t1_wa = wa_losses[0]
        t2_wa = wa_losses[1]
        out = (wa_losses_mn, t1_wa, t2_wa)
   
        return out


# MLM model
class MLMRetrieval(nn.Module):
    def __init__(self):
        super(MLMRetrieval, self).__init__()
        self.learn_img      = LearnImages()
        self.learn_sum      = LearnSummaries()
        self.image_coord    = CoordNet()
        self.learn_tri      = LearnTriples()
        self.fc1 = torch.nn.Linear(args.coordDim+args.coordDim, args.coordDim)

    def forward(self, input, sum, triple):
        # input embeddings
        img_emb = self.learn_img(input)
        sum_emb = self.learn_sum(sum)
        tri_emb = self.learn_tri(triple)

        # coord embedding
        img_coord = self.image_coord(img_emb)
        txt_coord = self.image_coord(sum_emb)
        tri_coord = self.image_coord(tri_emb) 

        # task 2
        txt_coord_triple = torch.cat((txt_coord, tri_coord), 1)
        txt_coord_triple = self.fc1(txt_coord_triple)

        return [img_coord, txt_coord, txt_coord_triple]
