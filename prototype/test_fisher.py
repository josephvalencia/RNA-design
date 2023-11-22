import torch
from torch.distributions import Categorical,Independent
import math
import torch.nn.functional as F

def test_fisher():
    
    shape = (50,4)
    n_samples = 10000
    logits = torch.randn(*shape).requires_grad_(True)
    probs = F.softmax(logits,dim=1).requires_grad_(True) 
    dist = Independent(Categorical(probs=probs),1)
   
    saved = []
    for i in range(n_samples): 
        samples = dist.sample()
        log_probs = dist.log_prob(samples) 
        log_probs.backward(torch.ones_like(log_probs),inputs=probs,retain_graph=True) 
        per_sample_grads = probs.grad
        saved.append(per_sample_grads.reshape(-1)) 

    G = torch.stack(saved)
    cov = (G.T @ G) / n_samples
    print('full(COV)',cov) 
    print('diag(COV)',torch.diag(cov))
    print(probs) 
    print(f'inverse_probs={probs.reshape(-1).pow(-1)}')
    print(f'logits = {logits.reshape(-1)}')
    print(f'exp(logits) = {torch.exp(logits.reshape(-1))}')  
    
    ''' 
    U,S,Vh = torch.linalg.svd(G.T,full_matrices=False)
    S_inv = S.pow(-2) 
    print(S_inv) 
    G_inv = Vh.T * S_inv @ Vh
    print(G_inv)
    '''

if __name__ == "__main__":

    test_fisher()
    

