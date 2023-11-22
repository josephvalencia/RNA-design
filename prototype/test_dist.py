import torch
from torch.distributions import Categorical,Independent,Dirichlet
import torch.nn.functional as F

logits = torch.randn(size=(50,4)).requires_grad_(True)
probs = F.softmax(logits,dim=0)
#dist = Independent(Categorical(probs=probs),0)
dist = Independent(Dirichlet(concentration=probs),1)
n_samples = 50000
storage = []
for i in range(n_samples):
    sample = dist.sample()
    sample_log_prob = dist.log_prob(sample)
    score = torch.autograd.grad(sample_log_prob,logits,retain_graph=True)[0].reshape(-1,1)
    norm = lambda x : torch.sqrt(torch.sum(x**2))
    outer_prod = score @ score.T 
    storage.append(outer_prod)

fisher_estimate = torch.stack(storage,dim=0).mean(dim=0)
print(f'MC = {fisher_estimate}')
analytical = torch.diag(torch.exp(logits.reshape(-1)))
print(f'analytical = {analytical}')
