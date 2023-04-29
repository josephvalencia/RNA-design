import torch
from typing import Callable, Optional,Union, Tuple, List
import torch.nn as nn
from utils.categorical import DifferentiableCategorical
import matplotlib.pyplot as plt

class OracleMaximizer:

    def __init__(self, oracles : List[Union[Callable,nn.Module]],
                loss_items : List[Tuple[Union[Callable,None],Union[torch.tensor,None],torch.tensor]],
                optimizer : Optional[torch.optim.Optimizer] = None):

        self.oracles = oracles
        self.loss_items = loss_items 
        self.optimizer = optimizer

    def preds_and_composite_loss(self,sequence):
        '''Evaluate all oracles on a sequence and return predictions and a weighted but unreduced loss'''
        total_loss = 0.0
        preds = []
        for oracle,(loss,target,weight) in zip(self.oracles,self.loss_items):
            pred = oracle(sequence)
            preds.append(pred.detach())
            if loss is not None and target is not None: 
                # a loss is something to minimize 
                total_loss += weight*loss(pred,target)
            else:
                # a scalar output is something to maximize, so we negate it
                total_loss += -weight*pred
        return preds, total_loss

    def fit(self,
            max_iter : int = 10000,
            convergence_tol : float = 1e-3,
            stalled_tol : int = 500,
            verbose : bool = False,
            report_interval : int = 100):

        print('Fitting OracleMaximizer')
        stalled_counter = 0
        device = 'cuda:0'
        seed_sequence = torch.randint(0,4,(1,50)).to(device)
        onehot_seed = encode_seq(seed_sequence)
        driver = DifferentiableCategorical(onehot_seed,grad_method='softmax',normalize_method=None) 
        driver.to(device)
        optimizer = torch.optim.Adam(driver.parameters(),lr=1e-3) 
        #optimizer = torch.optim.LBFGS(driver.parameters(),lr=1e-2) 
        initial_preds,initial_loss = self.preds_and_composite_loss(onehot_seed)
        best_loss = initial_loss
        best_seq = seed_sequence 
        print(f'Initial loss = {initial_loss.item()}')
        results = [initial_loss.detach().item()]
         
        for i in range(max_iter):

            next_sequence = driver.sample()
            next_preds,next_loss = self.preds_and_composite_loss(next_sequence)
            
            # perform the optimization step
            optimizer.zero_grad()
            next_loss.backward()

            def closure():
                optimizer.zero_grad()
                next_preds,next_loss = self.preds_and_composite_loss(next_sequence)
                next_loss.backward()
                return next_loss

            #optimizer.step(closure)
            optimizer.step()
            if i % report_interval == 0: 
                dense = next_sequence.argmax(dim=1).squeeze()
                diff = torch.count_nonzero(dense - seed_sequence)
                print(f'i={i},pred={next_preds},diff={diff.item()}')
                results.append(next_loss.detach().item())
            
            if next_loss < best_loss:
                best_loss = next_loss
                best_seq = next_sequence
                stalled_counter = 0
            else:
                stalled_counter
            if stalled_counter > stalled_tol:
                break
            
        plt.plot(results)
        plt.savefig('results.png')
        plt.close()
        
def encode_seq(seq):
    utr = torch.nn.functional.one_hot(seq,num_classes=4).permute(0,2,1).float().requires_grad_(True)
    return utr

def input_grads(inputs,outputs):
    ''' per-sample gradient of outputs wrt. inputs '''
    total_pred = outputs.sum()
    total_pred.backward(torch.ones_like(total_pred),inputs=inputs)
    grads = inputs.grad
    return grads