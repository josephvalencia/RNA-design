import torch
from typing import Callable, Optional, Union, Tuple, List
import torch.nn as nn
from seqopt.categorical import DifferentiableCategorical
from seqopt.mcmc import LangevinSampler, PathAuxiliarySampler, GibbsWithGradientsSampler, RandomSampler, SimulatedAnnealingSampler
from seqopt.ngd import NaturalGradientDescent
from tqdm import tqdm
from biotite.sequence.io.fasta import FastaFile
import time
from seqopt.oracle import NucleotideDesigner

class OracleMaximizer:

    def __init__(self,
                oracle : NucleotideDesigner, 
                seed_sequence : torch.tensor, 
                loss_items : List[Tuple[Union[Callable,None],Union[torch.tensor,None],torch.tensor]],
                device : str = "cpu", 
                mode : str = "optimize",
                mcmc : str = "langevin",
                grad : str = "normal",
                optimizer : Union[str,None] = None,
                max_iter : Optional[int] = None,
                mask_rare_tokens : bool = False,
                learning_rate : int = 1e-3,
                norm='instance',
                n_samples=1):
    
        self.seed_sequence = seed_sequence.to(self.device)
        self.oracles = oracle.evaluate
        self.onehot_fn = oracle.onehot_encode
        self.to_nucleotide = oracle.dense_decode
        self.class_dim = oracle.class_dim
        self.num_classes = oracle.num_classes 
        self.loss_items = loss_items
        self.device = device
        self.mode = mode
        self.mcmc = mcmc
        self.grad = grad
        self.learning_rate = learning_rate

        if self.mode == 'optimize': 
            self.driver = DifferentiableCategorical(
                                    self.seed_sequence,
                                    self.class_dim, 
                                    self.onehot_fn, 
                                    grad_method=grad,
                                    n_samples=n_samples,
                                    normalize_method=norm,
                                    mask_rare_tokens=mask_rare_tokens)
            self.driver.to(self.device)
        elif self.mode == 'sample':
            if mcmc == 'langevin':
                self.driver =  LangevinSampler(
                                    self.seed_sequence,
                                    num_classes=self.num_classes,
                                    class_dim=self.class_dim,
                                    output_fn=self.preds_and_composite_loss,
                                    onehot_fn=self.onehot_fn,
                                    stepsize=self.learning_rate)
            elif mcmc == 'path_auxiliary': 
                self.driver = PathAuxiliarySampler(
                                    self.seed_sequence,
                                    num_classes=self.num_classes,
                                    class_dim=self.class_dim,
                                    output_fn=self.preds_and_composite_loss,
                                    onehot_fn=self.onehot_fn)
            elif mcmc == 'gibbs_with_gradients':
                self.driver = GibbsWithGradientsSampler(
                                    self.seed_sequence,
                                    num_classes=self.num_classes,
                                    class_dim=self.class_dim,
                                    output_fn=self.preds_and_composite_loss,
                                    onehot_fn=self.onehot_fn) 
            elif mcmc == 'random':
                self.driver = RandomSampler(
                                    self.seed_sequence,
                                    num_classes=self.num_classes,
                                    class_dim=self.class_dim,
                                    output_fn=self.preds_and_composite_loss,
                                    onehot_fn=self.onehot_fn)
            elif mcmc == 'annealing':
                self.driver = SimulatedAnnealingSampler(
                                    self.seed_sequence,
                                    num_classes=self.num_classes,
                                    class_dim=self.class_dim,
                                    output_fn=self.preds_and_composite_loss,
                                    onehot_fn=self.onehot_fn,
                                    num_steps=max_iter)

        else:
            raise ValueError("mode must be one of optimize|sample")
        
        self.optimizer = None
        if mode == 'optimize': 
            if optimizer == 'adam': 
                self.optimizer = torch.optim.Adam(self.driver.parameters(),lr=self.learning_rate) 
            elif optimizer == 'lbfgs':  
                self.optimizer = torch.optim.LBFGS(self.driver.parameters(),lr=self.learning_rate) 
            elif optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(self.driver.parameters(),lr=self.learning_rate)
            elif optimizer == 'NGD':
                self.optimizer = NaturalGradientDescent(self.driver.parameters(),lr=self.learning_rate)
            else:
                raise ValueError("When `mode`==`optimize`,`optimizer` must be one of adam|lbfgs|SGD")

    def preds_and_composite_loss(self,sequence):
        '''Evaluate all oracles on a sequence and return a weighted loss'''
        total_loss = 0.0
        for oracle,(loss,target,weight) in zip(self.oracles,self.loss_items):
            pred = oracle(sequence)
            if loss is not None and target is not None: 
                # a loss is something to minimize 
                total_loss += weight*loss(pred,target)
            else:
                # a scalar output is something to maximize, so we negate it
                #print(f'negating once')
                total_loss += -weight*pred
        # Pytorch optimizers assume minimization, MCMC assumes maximization
        # so a final negation for MCMC
        if self.mode == 'sample':
            total_loss = -total_loss 
        #return total_loss.mean()
        #print(f'self.mode = {self.mode}, self.loss = {total_loss.mean()}')
        return total_loss
     
    def optimize_step(self,sequence):
            
        def closure():
            self.optimizer.zero_grad()
            if self.grad == 'reinforce':
                #values = self.preds_and_composite_loss(sequence)
                values = self.preds_and_composite_loss(sequence).unsqueeze(0)
                #values = self.preds_and_composite_loss(sequence).squeeze()
                print(f'vals = {values.shape}')
                print(f'log_prob = {self.driver.log_prob(sequence).shape}')
                #next_loss = torch.mean(values.squeeze(1) * self.driver.log_prob(sequence))
                next_loss = torch.mean(values * self.driver.log_prob(sequence))
                #with torch.autograd.profiler.profile(use_cuda=True) as prof:
                next_loss.backward()
                #print(prof.key_averages().table(sort_by="self_cuda_time_total"))
                return values.mean()
            else:
                next_loss = self.preds_and_composite_loss(sequence)
                next_loss = next_loss.mean() 
                next_loss.backward()
                return next_loss
        
        if isinstance(self.optimizer,torch.optim.LBFGS):
            self.optimizer.step(closure)
            loss = closure()
        elif isinstance(self.optimizer,NaturalGradientDescent):
            self.optimizer.zero_grad()
            vals = self.preds_and_composite_loss(sequence)
            if self.grad == 'reinforce':
                loss = torch.mean(vals.squeeze(1)*self.driver.log_prob(sequence))
            else:
                loss = loss.mean()

            #loss.backward(torch.ones_like(loss),inputs=(sequence,*self.driver.parameters()),retain_graph=True)
            loss.backward(retain_graph=True)
            # compute f'(x) for each x in sequence 
            sequence.grad = None
            vals.backward(torch.ones_like(vals),inputs=sequence)
            per_sample_grads = sequence.grad
            self.optimizer.step(per_sample_grads) 
        else: 
            loss = closure()
            ''' 
            l2_norm = lambda x : torch.sqrt(x.reshape(-1).pow(2).sum())
            for name,param in self.driver.named_parameters():
                if name == 'logits': 
                    to_print = 'grad is none' if param.grad is None else f'grad norm is {l2_norm(param.grad)}' 
                    print(to_print)
            '''
            self.optimizer.step()
        return loss

    def sample_step(self,sequence):
        pass

    def fit(self,
            max_iter : int = 20000,
            stalled_tol : int = 1000,
            report_interval : int = 10):

        # setup the seed sequence
        onehot_seed = self.onehot_fn(self.seed_sequence).to(self.device)
        readable = self.to_nucleotide(self.seed_sequence)

        # get initial predictions and loss
        with torch.no_grad():
            initial_loss = self.preds_and_composite_loss(onehot_seed)
        
        best_loss = initial_loss
        best_seq = self.onehot_fn(self.seed_sequence)
        
        if not self.mode == 'optimize':
            initial_loss *= -1
        results = [initial_loss.detach().item()]

        print(f'initial results: {results}')
        print(f'Fitting OracleMaximizer using {self.driver}')
        print(f'Seed sequence {readable} loss = {initial_loss.item():.3f}')
        stalled_counter = 0
        pbar = tqdm(range(max_iter))
        #pbar = range(max_iter)
        to_save = FastaFile()
        for i in pbar:
            # sample and perform gradient-based updates
            next_sequence = self.driver.sample()
            if self.mode == 'optimize': 
                next_loss = self.optimize_step(next_sequence)
            else:
                next_loss = -self.preds_and_composite_loss(next_sequence)
            # report progress
            next_dense = next_sequence.argmax(dim=self.class_dim)
            nucleotides = self.to_nucleotide(next_dense)
            header = f'step={i}'
            to_save[header] = nucleotides 
            results.append(next_loss.detach().item())
            # monitor convergence
            if next_loss < best_loss:
                print(f'next_loss {next_loss} is better than best_loss {best_loss}')
                best_loss = next_loss
                best_seq = next_sequence
                stalled_counter = 0
            else:
                stalled_counter += 1
            if i % report_interval == 0: 
                dense = best_seq.argmax(dim=self.class_dim).squeeze()
                #dense = best_seq.argmax(dim=self.class_dim).squeeze(2).permute(1,0)
                diff = torch.count_nonzero(dense.squeeze() != self.seed_sequence.squeeze(),dim=0)
                mean_diff = torch.sum(diff) / diff.numel()
            if stalled_counter > stalled_tol:
                print(f'Stopping early at iteration {i}')
                break
            #print(f'i={i}, best_loss = {best_loss}') 
            pbar.set_postfix({'best_loss': best_loss.item(),'diff': mean_diff.item()})
        
        to_save.write('trajectory.fasta')

        best_seq_dense = best_seq.argmax(dim=self.class_dim).squeeze()
        improvement = best_loss - initial_loss 
        print(f'best results: {best_loss}, improvement: {improvement}') 
        return best_seq_dense.detach(), improvement.detach(), results
