from abc import ABC, abstractmethod
import torch
#from distribution import DifferentiableCategorical
from typing import Union
import torch.nn.functional as F
import random

class DiscreteMCMCSampler():
    ''' Parent class for MCMC sampler from a discrete distribution using a gradient-informed proposal'''

    def __init__(self, start_state : torch.tensor ,
                    output_fn,
                    grad_fn,
                    metropolis_adjust=True,
                    constraint_functions=[]):
        
        self.curr_state = start_state
        self.output_fn = output_fn
        self.grad_fn = grad_fn
        self.metropolis = metropolis_adjust
        self.constraint_functions = constraint_functions
        self.step = 0
    
    def sample(self):
        
        # calc q(x'|x)
        q_fwd = self.forward_proposal_dist(src=self.curr_state)
        next_state = q_fwd.sample() 

        # metropolis adjustment
        if self.metropolis:
            curr_state = self.metropolis_adjustment(q_fwd,next_state)
        else:
            curr_state = next_state

        # apply constraints
        if len(self.constraint_functions):
            for constraint_fn in self.constraint_functions:
                curr_state = constraint_fn(curr_state)

        self.curr_state = curr_state
        self.step += 1
        return self.curr_state

    def metropolis_adjustment(self,q_fwd,next_state):
        # calc q(x|x')
        q_rev = self.reverse_proposal_dist(src=self.curr_state)
        # compute the acceptance probabilities and accept/reject 
        fwd_log_prob = q_fwd.log_prob(next_state)
        rev_log_prob = q_rev.log_prob(self.curr_state)
        score_diff = self.output_fn(next_state) - self.output_fn(self.curr_state)
        accept_probs = self.acceptance_probs(self,score_diff,fwd_log_prob,rev_log_prob) 
        return self.accept_reject(accept_probs,next_state,self.curr_state)

    def acceptance_probs(self,score_diff,forward_log_prob,reverse_log_prob):
        acceptance_log_prob = score_diff + reverse_log_prob - forward_log_prob
        return torch.minimum(acceptance_log_prob,torch.tensor([0.0],device=score_diff.device))
    
    def input_grads(self,src):
        '''Calculate the gradient of output_fn with respect to the input''' 
        score = self.output_fn(src)
        grads = self.grad_fn(score,src)
        return grads
    
    def forward_proposal_dist(self,**kwargs) -> torch.distributions.Distribution:
        raise NotImplementedError
    
    def reverse_proposal_dist(self,**kwargs) -> torch.distributions.Distribution:
        raise NotImplementedError

    def accept_reject(self,accept_probs,next_state,curr_state) -> torch.tensor:
        ''' Accept/reject the next state based on the acceptance probabilities. The resampled positions may vary
        depending on the sampler.''' 
        raise NotImplementedError

class LangevinSampler(DiscreteMCMCSampler):
    '''Implements Zhang et al. ICML 2022 https://proceedings.mlr.press/v162/zhang22t.html'''

    def __init__(self,start_state,output_fn,grad_fn,stepsize=0.01,beta=0.999,epsilon=1e-8,metropolis_adjust=True):
        super().__init__(start_state,output_fn,grad_fn,metropolis_adjust)
        self.stepsize = stepsize
        self.beta = beta
        self.epsilon = epsilon
        self.curr_state = start_state 
        self.preconditioner = torch.new_zeros(self.curr_start,dtype=torch.float)
    
    def accept_reject(self,accept_probs,next_state,curr_state):
        ''' All positions are potentially resampled, so we need to accept/reject each position independently'''
        random_draw = torch.log(torch.rand(accept_probs.shape)) 
        acceptances = accept_probs > random_draw
        return torch.where(acceptances,next_state,curr_state)
    
    def forward_proposal_dist(self,src):
        grads = self.input_grads(src)
        self.update_preconditioner(grads)
        return self.langevin_proposal_dist(grads,src)
         
    def reverse_proposal_dist(self,src):
        grads = self.input_grads(src)
        return self.langevin_proposal_dist(grads,src)
    
    def langevin_proposal_dist(self,grads,onehot):
        # Taylor approx 
        grad_current_char = (grads * onehot).sum(dim=3)
        mutation_scores = grads - grad_current_char
        # stepsize term
        temperature_term = (1.0 - onehot) / (self.alpha * self.scaled_preconditioner()) 
        logits = 0.5 * mutation_scores - temperature_term 
        logits = self.mask_rare_tokens(logits)
        proposal_dist = torch.distributions.Categorical(logits=logits.squeeze(2) / 2)
        return torch.distributions.Independent(proposal_dist,1)

    def mask_rare_tokens(self,logits):
        '''Non-nucleotide chars may receive gradient scores, set them to a small number'''
        mask1 = torch.tensor([0,0,1,1,1,1,0,0],device=logits.device)
        mask2 = torch.tensor([-30,-30,0,0,0,0,-30,-30],device=logits.device)
        return logits*mask1 + mask2
    
    def update_preconditioner(self,grads):
        ''' Inspired by Adam diagonal preconditioner '''
        grads = grads.pow(2)
        diagonal = grads
        biased = self.beta*self.preconditioner + (1.0-self.beta)*diagonal 
        self.preconditioner = biased / (1.0 - self.beta**(self.step+1))

    def scaled_preconditioner(self):
        return self.preconditioner.sqrt() + self.epsilon

class PathAuxiliarySampler(DiscreteMCMCSampler):
    '''Implements Sun et al. ICLR 2022 https://openreview.net/pdf?id=JSR-YDImK95 '''

    def __init__(self,start_state,output_fn,grad_fn,metropolis_adjust=True,constraint_functions=[],max_path_length=5):
        super().__init__(start_state,output_fn,grad_fn,metropolis_adjust,constraint_functions)
        self.max_path_length = max_path_length
        self.current_length = 1
        self.src_cache = []

    def forward_proposal_dist(self,src):

        self.current_length = random.randint(1,self.max_path_length)
        self.src_cache.append(src)

        # gradients at initial state 
        grads = self.input_grads(src)
        log_prob = 0.0
        for i in range(self.current_length): 
            q_i = self.path_auxiliary_proposal_dist(src,grads)
            next_state = q_i.sample() 
            log_prob += q_i.log_prob(next_state) 
            src = self.update_seq(next_state,src)
            self.src_cache.append(src) 
        return log_prob

    def reverse_proposal_dist(self,src):

        # gradients at final state 
        grads = self.input_grads(self.src_cache[-1])
        log_prob = 0.0
        
        for i in range(self.current_length - 1,0,-1): 
            q_i = self.path_auxiliary_proposal_dist(src[i],grads)
            # log prob of cached previous state rather than sampled next state 
            prev_state = self.src_cache[i-1] 
            log_prob += q_i.log_prob(prev_state) 

        return log_prob

    def update_seq(self,next_state,curr_state):
        pos_idx = next_state / curr_state.shape[0]
        char_idx = next_state % curr_state.shape[1]
        # update character at position pos_idx
        curr_state[pos_idx,:] = F.one_hot(char_idx,curr_state.shape[1])

    def path_auxiliary_proposal_dist(self,src,grads):
        grad_current_char = (grads * src).sum(dim=3)
        mutation_scores = grads - grad_current_char
        # V*(L-1) -way softmax 
        return torch.distributions.Categorical(logits=mutation_scores.reshape(-1) / 2)

    def reverse_proposal_dist(self,grads,onehot):
        pass

    def accept_reject(self,accept_probs,next_state,curr_state):
        pass

class GibbsWithGradientsSampler(DiscreteMCMCSampler):
    '''Implements Grathwohl et al. ICML 2021 '''
    
    def forward_proposal_dist(self,src):
        return self.gwg_proposal_dist(self,src)        

    def reverse_proposal_dist(self,src):    
        return self.gwg_proposal_dist(self,src)

    def gwg_proposal_dist(self,src):
        # Taylor approx 
        grads = self.input_grads(src)
        grad_current_char = (grads * src).sum(dim=3)
        mutation_scores = grads - grad_current_char
        # V*(L-1) way softmax 
        return torch.distributions.Categorical(logits=mutation_scores.reshape(-1) / 2)

    def accept_reject(self,accept_probs,next_state,curr_state):
        
        random_draw = torch.log(torch.rand(accept_probs.shape)) 
        acceptances = accept_probs > random_draw
        
        if acceptances:
            # convert i from flat softmax to position and character indices
            pos_idx = next_state / self.curr_state.shape[0]
            char_idx = next_state % self.curr_state.shape[1]
            # update character at position pos_idx
            curr_state[pos_idx,:] = F.one_hot(char_idx,self.curr_state.shape[1])
        
        return curr_state
