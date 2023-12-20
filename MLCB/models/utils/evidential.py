import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import contextlib
from functorch import jvp, grad, vjp

def hvp(f, primals, tangents):
  return jvp(grad(f), primals, tangents)[1]

class NormalInverseGammaDistribution:

    def __init__(self,mu : torch.Tensor,
                 nu : torch.Tensor,
                 alpha : torch.Tensor,
                 beta : torch.Tensor):
        self.mu = mu
        self.nu = nu
        self.alpha = alpha
        self.beta = beta

    def to_rep(self) -> str:
        batch_size = self.mu.shape[0]
        batch_reps= [] 
        for b in range(batch_size):
            entry = f'NormInvGamma(mu={self.mu[b].item():.2e},nu={self.nu[b].item():.2e},alpha={self.alpha[b].item():.2e},beta={self.beta[b].item():.2e})'
            batch_reps.append(entry)
        return batch_reps 

    def detach(self):
        return NormalInverseGammaDistribution(self.mu.detach(),
                                              self.nu.detach(),
                                              self.alpha.detach(),
                                              self.beta.detach())
    @property
    def mean(self):
        return self.mu

    @property
    def aleatoric_uncertainty(self):
        return self.beta / (self.alpha - 1)
    
    @property
    def epistemic_uncertainty(self):
        return self.aleatoric_uncertainty / self.nu
    
    @property
    def normal_dist(self):
        return D.Normal(self.mu,self.aleatoric_uncertainty)

    @classmethod
    def kl_divergence(cls, p : 'NormalInverseGammaDistribution',
                       q : 'NormalInverseGammaDistribution',
                       reduction='batchmean'):
        '''Compute the KL divergence between two Normal Inverse Gamma distributions'''
        '''
        kl = 1/2*(p.alpha/p.beta) * (p.mu-q.mu)**2 * q.nu \
                + 1/2*(q.nu/p.nu - 1 - torch.log(q.nu)+torch.log(p.nu)) \
                + q.alpha*(torch.log(p.beta) - torch.log(q.beta)) \
                + torch.lgamma(q.alpha) - torch.lgamma(p.alpha) \
                + (p.alpha - q.alpha) * torch.digamma(p.alpha) \
                - (p.beta - q.beta) * p.alpha / p.beta
        ''' 
        '''
        l1 = 1/2*(p.alpha/p.beta) * (p.mu-q.mu)**2 * q.nu 
        l2 = 1/2*(q.nu/p.nu - 1 - torch.log(q.nu)+torch.log(p.nu)) 
        l3 = q.alpha*(torch.log(p.beta) - torch.log(q.beta)) 
        l4 = torch.lgamma(q.alpha) - torch.lgamma(p.alpha) 
        l5 = (p.alpha - q.alpha) * torch.digamma(p.alpha) 
        l6 = (p.beta - q.beta) * p.alpha / p.beta
        kl = l1+l2+l3+l4+l5-l6 
        '''
        kl_gaussian = D.kl_divergence(p.normal_dist,q.normal_dist)
        kl = kl_gaussian
        
        reduced =  kl.mean()
        if torch.any(torch.isnan(reduced)):
            #print('P',p)
            #print('Q',q)
            raise ValueError('KL divergence is NaN')
        '''
        if reduced.item() > 2.0:
            print('Large KL divergence!')
            batchsize = p.mu.shape[0]
            P = p.to_rep()
            Q = q.to_rep()
            for b in range(batchsize):
                print(f'P{b}: {P[b]}')
                print(f'Q{b}: {Q[b]}')
                print(f'KL[P|Q]: {kl[b].item()}, total: {reduced.item()}')
                #for i,row in enumerate([l1,l2,l3,l4,l5,l6]):
                #    print(f'l{i+1}: {row[b].item():.2e}')
        '''
        return reduced
    
class PseudoHuberLoss(torch.nn.Module):

    def __init__(self,delta=1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, error : torch.Tensor) -> torch.Tensor:
        '''Compute the Pseudo-Huber loss'''
        delta = self.delta
        return delta**2 * (torch.sqrt(1+error**2/delta**2)-1)

class EvidentialLoss(torch.nn.Module):

    def __init__(self,error_weight,reduction='sum',epsilon=1e-8):
        super().__init__()
        self.error_weight = error_weight
        self.epsilon = epsilon
        self.reduction = reduction
        self.pseudo_huber = PseudoHuberLoss()

    def forward(self, D : NormalInverseGammaDistribution,
                target : torch.Tensor) -> torch.Tensor:
        
        '''Compute the evidential loss for a regression problem'''

        error = target - D.mu
        omega = 2*D.beta*(1+D.nu)

        # negative log likelihood of the evidence (Student's t Dribution)
        fn = lambda x : torch.any(torch.isnan(x))
        nll = 1/2*torch.log(math.pi / D.nu) - D.alpha * torch.log(omega) \
                + (D.alpha + 1/2) * torch.log(omega + D.nu*error**2)  \
                + torch.lgamma(D.alpha) - torch.lgamma(D.alpha + 1/2) 
        
        # error regularization
        #error_loss = torch.abs(error) * (2*D.nu+D.alpha)
        error_loss = self.pseudo_huber(error)*(2*D.nu+D.alpha)
        loss = nll + self.error_weight * (error_loss - self.epsilon)
        
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss        

# LDS adapted from https://github.com/lyakaap/VAT-pytorch/

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


class ERVirtualAdversarialLoss(nn.Module):

    def __init__(self, exact=True, xi=1e-6, eps=0.01, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(ERVirtualAdversarialLoss, self).__init__()
        self.exact = exact 
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def l2_normalize(self, d):
        #return F.normalize(d.view(d.size(0), -1), dim=1).reshape(d.shape)
        normed = d / (torch.sqrt((d**2).sum(dim=(1,2),keepdim=True)) + 1e-8)
        check = torch.sqrt((normed**2).sum(dim=(1,2),keepdim=True))
        return normed 
    
    '''
    def l2_normalize(self,d):
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
        return d
    ''' 

    def forward(self, model, x):

        with torch.no_grad(): 
            pred = model(x).detach()
        
        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = self.l2_normalize(d)

        with torch.enable_grad():
            with _disable_tracking_bn_stats(model):
                # calc adversarial direction by power iteration
                for _ in range(self.ip):
                    if self.exact: # exact computation of Hd
                        def f(r):
                            perturbed = x+r
                            pred_hat = model(perturbed)
                            return NormalInverseGammaDistribution.kl_divergence(pred,pred_hat)
                        r_zeros = torch.zeros_like(d).requires_grad_(True)
                        d = hvp(f,(r_zeros,),(d,))
                        d = self.l2_normalize(d)
                        model.zero_grad() 
                    else: # finite differences from original paper
                        d.requires_grad_(True)
                        perturbed = x + self.xi*d
                        pred_hat = model(perturbed)
                        adv_distance = NormalInverseGammaDistribution.kl_divergence(pred,pred_hat)
                        adv_distance.backward()
                        d = self.l2_normalize(d.grad)
                        model.zero_grad()
        # calc LDS
        r_adv = self.eps*d.detach()
        pred_hat = model(x + r_adv)
        lds = NormalInverseGammaDistribution.kl_divergence(pred,pred_hat)
        print(f'LDS: {lds.item()}')
        return lds

class EvidentialRegressionOutputLayer(torch.nn.Module):

    def __init__(self,input_dim):
        super().__init__()
        self.output = torch.nn.Linear(input_dim,4)

    def forward(self,x):
        # output the Normal Inverse Gamma parameters 
        mu,nu,alpha,beta = torch.split(self.output(x),1,dim=1)
        nu = F.softplus(nu)
        alpha = F.softplus(alpha)+1
        beta = F.softplus(beta)
        return NormalInverseGammaDistribution(mu,nu,alpha,beta)
    
