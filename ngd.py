import torch

class NaturalGradientDescent(torch.optim.Optimizer):
    
    '''Natural gradient descent optimizer. Assumes that the parameters being estimated are
    independent categorical distributions. Uses the Fisher information matrix for this case and 
    adds a low rank update from a sample of gradients.'''

    def step(self):
        for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    natural_grad = self.natural_gradient(grad) 
                    p.data.add_(-self.lr * natural_grad)

    def natural_gradient(self,grads):
        ''' 11/08/2022. Pseudocode sketch of how to form diagonal + low rank estimate of inverse Fisher information 
            matrix and compute natural gradient descent direction. Currently handles up to 24 samples at typical
            sequence lengths for bioseq2seq. NGD time overhead is equivalent to ~1 sample worth, appears not to
            alter asymptotic time complexity because input gradient calculation dominates. Memory usage increases
            quadratically with params due to cost of instantiating FIM.
        '''

        N_SAMPLES = 24
        G = grads.reshape(N_SAMPLES,-1)
        # diagonalize with economy SVD
        U,S,Vh = torch.linalg.svd(G,full_matrices=False)
        # square to convert G singular vals to G^TG eigvals, then invert 
        S_inv = torch.diag(S.pow(-2)) 
        # placeholder for closed-form diagonal Fisher of independent categoricals
        diag_inv_col = G.new_ones(G.shape[1],1)

        # broadcast to avoid matmul with diagonal, ~order of magnitude faster
        V_scaled = diag_inv_col * Vh.T
        # only an N_SAMPLES x N_SAMPLES  matrix to invert
        inner_inv = torch.linalg.inv(S_inv + Vh @ V_scaled) 
        # Woodbury update of diagonal inverse with low rank term
        woodbury_inv_fisher = torch.diag(diag_inv_col.squeeze()) - V_scaled @ inner_inv @ V_scaled.T 
        # TODO: add special case for rank-1 updates with Sherman-Morrison, no SVD needed
        # use averaged grads as descent direction 
        avg_grad = G.mean(dim=0)
        # precondition to obtain NGD step
        natural_grad = woodbury_inv_fisher @ avg_grad
        # return to proper input shape
        natural_grad = natural_grad.reshape(*grads.shape)
        return natural_grad