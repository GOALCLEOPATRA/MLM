import torch
import torch.nn as nn
from args import get_parser


# read parser
parser = get_parser()
args = parser.parse_args()

def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

# embed wiki text - need to be changed
class Img2Vec(nn.Module):
    def __init__(self):
        super(Img2Vec, self).__init__()

    def forward(self, x):
        return x

# embed wiki text - need to be changed
class Wiki2Vec(nn.Module):
    def __init__(self):
        super(Wiki2Vec, self).__init__()
        self.lstm = nn.LSTM(input_size=args.wikiDim, hidden_size=args.embDim, bidirectional=False, batch_first=True)

    def forward(self, x):
        out, hidden = self.lstm(x.unsqueeze(1))
        return out.squeeze(1)

# embed coordinates
class Coord2Vec(nn.Module):
    def __init__(self):
        super(Coord2Vec, self).__init__()
        self.linear = nn.Linear(in_features=2, out_features=args.embDim)

    def forward(self, x):
        return self.linear(x)

# embed coordinates
class Triple2Vec(nn.Module):
    def __init__(self):
        super(Triple2Vec, self).__init__()
        self.linear = nn.Linear(in_features=2048, out_features=args.embDim)

    def forward(self, x):
        return self.linear(x)

# MLM model
class MLMRetrieval(nn.Module):
    def __init__(self):
        super(MLMRetrieval, self).__init__()
        self.img2vec        = Img2Vec()
        self.wiki2vec       = Wiki2Vec()
        self.coord2vec      = Coord2Vec()
        self.triple2vec     = Triple2Vec()
        self.mml_emp = torch.Tensor([False, False, False])
        self.n_mml = len(self.mml_emp)
        self.log_vars = torch.nn.Parameter(torch.zeros(self.n_mml))
        self.mse = nn.MSELoss()

        self.visual_embedding = nn.Sequential(
            nn.Linear(args.imgDim, args.embDim),
            nn.Tanh(),
        )

        self.wiki_embedding = nn.Sequential(
            nn.Linear(args.wikiDim, args.embDim),
            nn.Tanh(),
        )

        self.coord_embedding = nn.Sequential(
            nn.Linear(args.coordDim, args.embDim),
            nn.Tanh(),
        )

        self.triple_embedding = nn.Sequential(
            nn.Linear(args.tripleDim, args.embDim),
            nn.Tanh(),
        )

    def forward(self, img, en_wiki, fr_wiki, de_wiki, coord, triple):
        # wiki embedding
        enwiki_emb = self.wiki2vec(en_wiki)
        enwiki_emb = norm(enwiki_emb)

        # coord embedding
        coord_emb = self.coord2vec(coord)
        coord_emb = norm(coord_emb)

        # triple embedding
        triple_emb = self.triple2vec(triple)
        triple_emb = norm(triple_emb)

        # combined embedding
        comb_embed = torch.cat([enwiki_emb, coord_emb], 1) # joining on the last dim
        # recipe_emb = self.recipe_embedding(recipe_emb)
        # recipe_emb = norm(recipe_emb)

        # visual embedding
        visual_emb = self.img2vec(img)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        visual_emb = self.visual_embedding(visual_emb)
        visual_emb = norm(visual_emb)

        # mm losses
        coo_emb = torch.tanh(coord_emb)
        tri_emb = torch.tanh(triple_emb)
        enw_emb = torch.tanh(enwiki_emb)
        vis_emb = torch.tanh(visual_emb)
        #print('coo_emb_1', coo_emb)
        # pairwise l2 loss
        xc_e = self.mse(coo_emb, enw_emb)
        xv_e = self.mse(vis_emb, enw_emb)
        xc_v = self.mse(coo_emb, vis_emb)
        # weighted average
        mm_losses = torch.stack((xc_e, xv_e, xc_v))
        dtype = mm_losses.dtype
        device = mm_losses.device
        stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)
        weights = 1 / ((self.mml_emp.to(device).to(dtype)+1)*(stds**2))
        wa_losses = weights*mm_losses + torch.log(stds)
        ind_losses = wa_losses
        wa_ave_mn = wa_losses.mean()

        # final output
        output = [vis_emb, coo_emb, enw_emb, wa_ave_mn]

        return output