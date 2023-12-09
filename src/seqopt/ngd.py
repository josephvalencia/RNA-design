import torch
import math

class NaturalGradientDescent(torch.optim.SGD):
    
    '''Natural gradient descent optimizer. Assumes that the parameters being estimated are
    independent categorical distributions. Uses the Fisher information matrix for this case and 
    adds a low rank update from a sample of gradients.'''

    def step(self,per_sample_grads):
        for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        print(f'param = {p.name}, no gradient')
                        continue
                    grad = p.grad.data
                    a = per_sample_grads.mean(0).detach().cpu().numpy().ravel()
                    b = grad.detach().cpu().numpy().ravel()
                    natural_grad = self.natural_gradient(grad,per_sample_grads,p.data) 
                    p.data.add_(-group['lr'] * natural_grad)

    def natural_gradient(self,grads,per_sample_grads,params):
        ''' 11/08/2022. Pseudocode sketch of how to form diagonal + low rank estimate of inverse Fisher information 
            matrix and compute natural gradient descent direction. Currently handles up to 24 samples at typical
            sequence lengths for bioseq2seq. NGD time overhead is equivalent to ~1 sample worth, appears not to
            alter asymptotic time complexity because input gradient calculation dominates. Memory usage increases
            quadratically with params due to cost of instantiating FIM.
        '''
        N_SAMPLES = per_sample_grads.shape[0]
        G = per_sample_grads.reshape(N_SAMPLES,-1) / math.sqrt(N_SAMPLES)
        # diagonalize with economy SVD
        U,S,Vh = torch.linalg.svd(G,full_matrices=False)
        #S_inv = torch.diag(S.pow(-2)) 
        S_inv = S.pow(2) 
        # diagonal Fisher of independent categoricalsd
        ''' 
        diag_inv_col = torch.exp(-params).reshape(-1,1)
        print(diag_inv_col)
        # broadcast to avoid matmul with diagonal, ~order of magnitude faster
        V_scaled = diag_inv_col * Vh.T
        # only an N_SAMPLES x N_SAMPLES  matrix to invert
        inner_inv = torch.linalg.inv(S_inv + Vh @ V_scaled) 
        # Woodbury update of diagonal inverse with low rank term
        woodbury_inv_fisher = torch.diag(diag_inv_col.squeeze()) - V_scaled @ inner_inv @ V_scaled.T 
        # TODO: add special case for rank-1 updates with Sherman-Morrison, no SVD needed
        # use averaged grads as descent direction 
        frob_norm = lambda x : torch.sqrt(torch.sum(x.reshape(-1).pow(2)))
        wood_norm = frob_norm(woodbury_inv_fisher) 
        params_norm = frob_norm(params)
        diag_norm = frob_norm(diag_inv_col)
        cov_norm = frob_norm(cov)
        #print(f'S_inv shape = {S_inv.shape}, woodbury shape = {woodbury_inv_fisher.shape}')
        #print(f'diag norm = {diag_norm:.5f}, cov norm = {cov_norm:.5f}, S_inv norm = {frob_norm(S_inv):.5f}')
        # precondition to obtain NGD step
        ''' 
        frob_norm = lambda x : torch.sqrt(torch.sum(x.reshape(-1).pow(2)))
        batch_inv = Vh.T * S_inv @ Vh
        #print(f'batch_inv norm = {frob_norm(batch_inv):.5f}')
        #natural_grad = woodbury_inv_fisher @ grads.reshape(-1,1)
        natural_grad = batch_inv @ grads.reshape(-1,1) 
        #print(f"woodbury norm = {wood_norm:.5f},params norm = {params_norm:.5f}")
        #print(f"natural grad norm = {frob_norm(natural_grad):.5f}, grad norm = {frob_norm(grads):.5f}")
        # return to proper input shape
        natural_grad = natural_grad.reshape(*grads.shape)
        
        return natural_grad