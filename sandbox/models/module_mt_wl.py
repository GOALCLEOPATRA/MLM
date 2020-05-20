


import torch
import torch.nn as nn
import torch.optim as optim

# args
num_epochs = 10

# inputs
# range
r1 = -0.05
r2 = 0.05
# uniform in range
x1 = torch.FloatTensor(64, 2).uniform_(r1, r2)
x2 = torch.FloatTensor(64, 2).uniform_(r1, r2)

# targets
y1 = torch.tensor([-1., -1., -1., -1.,  1.,  1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1.,  1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,  1., -1., -1.,  1.,
        -1.,  1.,  1.,  1., -1., -1., -1.,  1., -1., -1., -1., -1., -1.,  1.,
        -1., -1.,  1., -1., -1.,  1., -1., -1.])
y2 = torch.tensor([ 1., -1., -1.,  1.,  1., -1., -1., -1., -1.,  1.,  1., -1.,  1., -1.,
         1., -1., -1., -1., -1., -1.,  1., -1.,  1., -1., -1., -1., -1., -1.,
        -1., -1.,  1., -1., -1., -1., -1.,  1., -1., -1.,  1., -1., -1.,  1.,
         1., -1., -1., -1.,  1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
        -1., -1.,  1., -1., -1., -1., -1., -1.])

# model
class MultiTaskLoss(nn.Module):
    def __init__(self, tasks):
        super(MultiTaskLoss, self).__init__()
        self.n_tasks = len(tasks)
        self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))

    def forward(self, losses):
        dtype = losses.dtype
        device = losses.device
        stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)
        weights = 1 / ((tasks.to(device).to(dtype)+1)*(stds**2))
        mt_losses = weights*losses + torch.log(stds)
        ind_losses = mt_losses
        mt_loss_mn = mt_losses.mean()
        
        return ind_losses, mt_loss_mn

# tasks
tasks = torch.Tensor([False, False])

# instantiate loss and model
cos_criterion = nn.CosineEmbeddingLoss(margin=0.1)
mtl = MultiTaskLoss(tasks) 

# optimizer
#optimizer = torch.optim.Adam(mtl.parameters(), lr = 0.1)
optimizer = optim.SGD(mtl.parameters(), lr = 0.1)

# train
for i in range(num_epochs):

    # tasks
    tl_a = cos_criterion(x1, x2, y1)
    tl_b = cos_criterion(x1, x2, y2)
    losses = torch.stack((tl_a, tl_b))
    
    optimizer.zero_grad()
    mtl_out = mtl(losses)
    in_losses = mtl_out[0]
    print('in_losses', in_losses)
    mt_loss = mtl_out[1]
    print('mt_loss', mt_loss)
    mt_loss.backward()
    optimizer.step()