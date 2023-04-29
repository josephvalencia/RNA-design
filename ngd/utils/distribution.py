from torch.distributions import Categorical, Independent
import torch.nn as nn

''' Wraps torch.distributions.Categorical to make it differentiable.
It will accomodate sampling from the discrete distribution and computing derivatives with respect to the logits.
This will be used for activation maximization to optimize the sequence for a particular oracle function.'''

class DifferentiableCategorical(nn.Module):

    def __init__(self,logits,grad_method='score_function'):
                
        self.logits = logits
        self.dist = Independent(Categorical(logits=logits.squeeze(2)),1)
        self.grad_method = grad_method 
    
    def sample(self):
        return self.dist.sample()

    def sample_n(self):
        return self.dist.sample_n()

    def log_prob(self):
        return self.dist.log_prob()

    def params(self):
        # return the parameters of the distribution
        return self.logits

    def update_logits(self,new_logits):
        self.logits = new_logits
        self.dist = Independent(Categorical(logits=new_logits.squeeze(2)),1)