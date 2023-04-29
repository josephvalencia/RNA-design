from torch.distributions import Categorical, Exponential, Independent
import torch
import torch.nn.functional as F
import torch.nn as nn

''' Wraps torch.distributions.Categorical to make it differentiable.
It will accomodate sampling from the discrete distribution and computing derivatives with respect to the logits.
This will be used for activation maximization to optimize the sequence for a particular oracle function.'''

class DifferentiableCategorical(nn.Module):

    def __init__(self,seed_sequence,grad_method='straight_through',normalize_method=None):

        super().__init__()
        self.seed_sequence = seed_sequence 
        self.logits = nn.Parameter(self.init_logits(seed_sequence))
        self.grad_method = grad_method

        if normalize_method == 'instance':
            self.normalize = nn.InstanceNorm1d(num_features=self.logits.shape[1],affine=True)
        elif normalize_method == 'layer':
            self.normalize = nn.LayerNorm()
        else:
            self.normalize = nn.Identity()

    def init_logits(self,onehot,true_val=5,other_val=0.1):
        ''' Initialize logits based on a relaxation of the seed sequence one-hot encoding'''
        logits = torch.where(onehot == 1,true_val,other_val) 
        return logits

    def sample(self):
        return self.__call__()

    def forward(self):
        # normalize logits as in Linder and Seelig 2021 https://doi.org/10.1186/s12859-021-04437-5
        logits = self.normalize(self.logits) 
        if self.grad_method == 'normal':
            return straight_through_sample(logits)
        elif self.grad_method == 'softmax':
            return softmax_straight_through_sample(logits)
        elif self.grad_method == 'gumbel_softmax':
            return gumbel_softmax_straight_through_sample(logits)
        elif self.grad_method == 'gumbel_rao_softmax':
            return gumbel_rao_softmax_straight_through_sample(logits)

    def __dist__(self):
        logits = self.normalize(self.logits) 
        return Independent(Categorical(logits=logits.permute(0,2,1)),1)
    
    def sample_n(self,n):
        return self.__dist__().sample((n,))

    def log_prob(self,sample):
        return self.__dist__().log_prob(sample)

    def probs(self):
        logits = self.normalize(self.logits) 
        return F.softmax(logits,dim=1)
     
class ForwardBackwardWrapper(torch.autograd.Function):
    ''' Trick from Thomas Viehmann https://discuss.pytorch.org/t/relu-with-leaky-derivative/32818/10.
    Behaves like x_forward on the forward pass, and like x_backward on the backward pass.'''
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward
    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

def to_onehot(x,num_classes):
    ''' Converts a tensor of integers to a one-hot encoding. '''
    onehot = F.one_hot(x,num_classes=num_classes).permute(0,2,1).float().requires_grad_(True)
    return onehot

def straight_through_sample(logits):
    '''Original STE from Bengio et. al 2013 http://arxiv.org/abs/1308.3432'''
    logits = logits.permute(0,2,1) 
    dist = Independent(Categorical(logits=logits),0)
    sample = dist.sample()
    onehot_sample = to_onehot(sample,num_classes=logits.shape[-1])
    logits = logits.permute(0,2,1)
    # 1 in the position of the sample, 0 everywhere else
    surrogate = torch.zeros_like(logits)
    surrogate.scatter_(2,sample.unsqueeze(2),1)
    return ForwardBackwardWrapper.apply(onehot_sample,surrogate)

def softmax_straight_through_sample(logits,temperature=1.0):
    '''Softmax STE from Chung et. al 2017 http://arxiv.org/abs/1609.01704'''
    logits = logits.permute(0,2,1) 
    dist = Independent(Categorical(logits=logits),0)
    sample = dist.sample()
    onehot_sample = to_onehot(sample,num_classes=logits.shape[-1])
    logits = logits.permute(0,2,1)
    normalized_logits = F.softmax(logits / temperature,dim=2)
    return ForwardBackwardWrapper.apply(onehot_sample,normalized_logits)

def gumbel_softmax_straight_through_sample(logits,temperature=0.1,hard=False):
    '''Gumbel softmax STE from Jang et. al 2017 http://arxiv.org/abs/1611.01144'''
    logits = logits.permute(0,2,1) 
    dist = Independent(Categorical(logits=logits),0)
    sample = dist.sample()
    onehot_sample = to_onehot(sample,num_classes=logits.shape[-1])
    logits = logits.permute(0,2,1)
    gumbel_softmax_logits = F.gumbel_softmax(logits,tau=temperature,hard=False,dim=2)
    return ForwardBackwardWrapper.apply(onehot_sample,gumbel_softmax_logits)

def gumbel_rao_softmax_straight_through_sample(logits,temperature=0.1,n_samples=1):
    '''Rao-Blackwellized Gumbel softmax STE from Paulus et. al 2020 http://arxiv.org/abs/2010.04838
    with code taken from https://github.com/nshepperd/gumbel-rao-pytorch/blob/master/gumbel_rao.py'''
    logits = logits.permute(0,2,1) 
    dist = Independent(Categorical(logits=logits),0)
    sample = dist.sample()
    onehot_sample = to_onehot(sample,num_classes=logits.shape[-1])
    logits = logits.permute(0,2,1)
    adjusted_logits = logits + conditional_gumbel(logits, onehot_sample, k=n_samples)
    gumbel_softmax_logits = F.softmax(adjusted_logits / temperature,dim=2).mean(dim=0)
    return ForwardBackwardWrapper.apply(onehot_sample,gumbel_softmax_logits)

@torch.no_grad()
def conditional_gumbel(logits, D, k=1):
    """Outputs k samples of Q = StandardGumbel(), such that argmax(logits
    + Q) is given by D (one hot vector)."""
    # iid. exponential
    E = Exponential(rate=torch.ones_like(logits)).sample([k])
    # E of the chosen class
    Ei = (D * E).sum(dim=-1, keepdim=True)
    # partition function (normalization constant)
    Z = logits.exp().sum(dim=-1, keepdim=True)
    # Sampled gumbel-adjusted logits
    adjusted = (D * (-torch.log(Ei) + torch.log(Z)) +
                (1 - D) * -torch.log(E/torch.exp(logits) + Ei / Z))
    return adjusted - logits













