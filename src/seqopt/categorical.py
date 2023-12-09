from torch.distributions import Categorical, Exponential, Independent
import torch
import torch.nn.functional as F
import torch.nn as nn

''' Wraps torch.distributions.Categorical to make it differentiable.
It will accomodate sampling from the discrete distribution and computing derivatives with respect to the logits.
This will be used for activation maximization to optimize the sequence for a particular oracle function.'''

class ReshapedInstanceNorm1d(nn.Module):
    ''' Reshape the input to 2D before applying InstanceNorm1d '''
    def __init__(self,num_features,class_dim,affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.class_dim = class_dim
        self.instance_norm = nn.InstanceNorm1d(num_features=self.num_features,affine=self.affine)

    def forward(self,x):
        
        dim_order = [self.class_dim] + [x for x in range(x.dim()) if x != self.class_dim]
        restored_order = [] 
        # put back in the original order  
        for i in range(x.dim()):
            if i < self.class_dim:
                restored_order.append(i+1)
            elif i == self.class_dim:
                restored_order.append(0)
            else:
                restored_order.append(i)
        x = x.permute(*dim_order)
        old_shape = x.shape 
        x = x.reshape(self.num_features,-1)
        x = self.instance_norm(x).reshape(*old_shape)
        x = x.permute(*restored_order)
        return x

class DifferentiableCategorical(nn.Module):

    def __init__(self,seed_sequence,class_dim,onehot_fn,grad_method='straight_through',n_samples=1,normalize_method=None,mask_rare_tokens=False):

        super().__init__()
        self.seed_sequence = seed_sequence
        self.class_dim = class_dim
        self.to_onehot = onehot_fn
        self.logits = nn.Parameter(self.init_logits(self.to_onehot(seed_sequence)))
        self.grad_method = grad_method
        self.mask = mask_rare_tokens
        self.n_samples = n_samples
        self.step = 0
        if normalize_method == 'instance':
            self.normalize = ReshapedInstanceNorm1d(num_features=self.logits.shape[self.class_dim],
                                                    class_dim = self.class_dim,affine=True)
        elif normalize_method == 'layer':
            self.normalize = nn.LayerNorm(normalized_shape=self.logits.size())
        else:
            self.normalize = nn.Identity()


    def init_logits(self,onehot,true_val=5.0,other_val=0.1):
        ''' Initialize logits based on a relaxation of the seed sequence one-hot encoding'''
        logits = torch.where(onehot == 1,true_val,other_val)
        return logits

    def mask_rare_tokens(self,logits):
        '''Non-nucleotide chars may receive gradient scores, set them to a small number'''
        mask1 = torch.tensor([0,0,1,1,1,1,0,0],device=logits.device)
        mask2 = torch.tensor([-10000,-10000,0,0,0,0,-10000,-10000],device=logits.device)
        return logits*mask1 + mask2
    
    def onehot_sample(self):
        #sample = self.__dist__().sample()
        #sample = self.sample_n(self.n_samples).squeeze(1) #.transpose(1,0) #.squeeze(2)
        sample = self.sample_n(self.n_samples).transpose(1,0).squeeze(2)
        onehot_sample = self.to_onehot(sample)
        return onehot_sample

    def sample(self):
        return self.__call__()

    def forward(self):
        # normalize logits as in Linder and Seelig 2021 https://doi.org/10.1186/s12859-021-04437-5
        sampled = self.onehot_sample()
        self.step+=1
        logits = self.normalize(self.logits) 
        if self.mask: 
            logits = self.mask_rare_tokens(logits)
        # differentiable sampling 
        if self.grad_method == 'normal':
            surrogate = straight_through_surrogate(logits,sampled)
            return ForwardBackwardWrapper.apply(sampled,surrogate)
        elif self.grad_method == 'softmax':
            surrogate = softmax_straight_through_surrogate(logits,sampled)
            return ForwardBackwardWrapper.apply(sampled,surrogate)
        elif self.grad_method == 'gumbel_softmax':
            surrogate = gumbel_softmax_straight_through_surrogate(logits,sampled)
            return ForwardBackwardWrapper.apply(sampled,surrogate)
        elif self.grad_method == 'gumbel_rao_softmax':
            surrogate = gumbel_rao_softmax_straight_through_surrogate(logits,sampled)
            return ForwardBackwardWrapper.apply(sampled,surrogate)
        elif self.grad_method == 'reinforce':
            return sampled
        
    def __dist__(self):
        logits = self.normalize(self.logits) 
        if self.mask: 
            logits = self.mask_rare_tokens(logits)
        dim_order = [x for x in range(logits.dim()) if x != self.class_dim] + [self.class_dim]
        return Independent(Categorical(logits=logits.permute(*dim_order)),1)
    
    def sample_n(self,n):
        return self.__dist__().sample((n,))

    def log_prob(self,onehot_sample):
        dense_sample = torch.argmax(onehot_sample,dim=self.class_dim)
        return self.__dist__().log_prob(dense_sample)

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

def straight_through_surrogate(logits,onehot_sample):
    '''Original STE from Bengio et. al 2013 http://arxiv.org/abs/1308.3432'''
    return logits * onehot_sample

def softmax_straight_through_surrogate(logits,onehot_sample,temperature=1.0):
    '''Softmax STE from Chung et. al 2017 http://arxiv.org/abs/1609.01704'''
    return  F.softmax(logits / temperature,dim=2)

def gumbel_softmax_straight_through_surrogate(logits,onehot_sample,temperature=10,hard=False):
    '''Gumbel softmax STE from Jang et. al 2017 http://arxiv.org/abs/1611.01144'''
    return F.gumbel_softmax(logits,tau=temperature,hard=False,dim=2) 

def gumbel_rao_softmax_straight_through_surrogate(logits,onehot_sample,temperature=0.1,n_samples=64):
    '''Rao-Blackwellized Gumbel softmax STE from Paulus et. al 2020 http://arxiv.org/abs/2010.04838
    with code taken from https://github.com/nshepperd/gumbel-rao-pytorch/blob/master/gumbel_rao.py'''
    adjusted_logits = logits + conditional_gumbel(logits, onehot_sample, k=n_samples)
    return F.softmax(adjusted_logits / temperature,dim=2).mean(dim=0)

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
