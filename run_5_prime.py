import lightning.pytorch as pl
from models.optimus_5prime.train_5_prime import MeanRibosomeLoadModule# OptimusFivePrime
from utils.mcmc import LangevinSampler, GibbsWithGradientsSampler, PathAuxiliarySampler
from utils.oracle_maximizer import OracleMaximizer
import torch
import matplotlib.pyplot as plt

def encode_seq(seq):
    utr = torch.nn.functional.one_hot(seq,num_classes=4).permute(0,2,1).float().requires_grad_(True)
    return utr

'''
if __name__ == "__main__":

    module = MeanRibosomeLoadModule.load_from_checkpoint(
        "models/optimus_5prime/checkpoints/epoch=2-val_loss=0.1396.ckpt")

    model = module.model
    model.eval()

    random_seq = torch.randint(0,4,(1,50))
    score = model(encode_seq(random_seq))
    print(f'Initial score = {score.item()}')
    
    sampler = LangevinSampler(random_seq,4,model,stepsize=0.05,beta=0.999,epsilon=1e-5)
    #sampler = GibbsWithGradientsSampler(random_seq,4,model)
    #sampler = PathAuxiliarySampler(random_seq,4,model)
    initial_score = model(encode_seq(random_seq)) 
    results = [initial_score.item()]
    for i in range(10000):
        seq = sampler.sample()
        if i % 100 == 0:
            score = model(encode_seq(seq))
            diff = torch.count_nonzero(seq - random_seq)
            print(f'Iteration {i}: {score.item()}, diff = {diff.item()}')
            results.append(score.item())

   
'''

if __name__ == "__main__":

    module = MeanRibosomeLoadModule.load_from_checkpoint("models/optimus_5prime/checkpoints/epoch=2-val_loss=0.1396.ckpt")
    model = module.model
    model.eval()
    model.to('cuda') 
    print('MRL model loaded')
    maximizer = OracleMaximizer([model],[(None,None,1.0)])
    maximizer.fit()



