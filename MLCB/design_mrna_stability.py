import torch
from models.saluki.model import SalukiDegradation
from seqopt.oracle import NucleotideDesigner
from trials import parse_args, setup_model_from_lightning, lower_confidence_bound,run_all_trials
from functools import partial

class StabilityDesigner(NucleotideDesigner):
        
        def __init__(self,forward_fn):
            super().__init__(num_classes=6,class_dim=2)
            self.fwd = forward_fn

        def onehot_encode(self,seq):
            '''Index tensor to one-hot encoding'''
            seq = torch.nn.functional.one_hot(seq,num_classes=6).float()
            length = seq.shape[1]
            aux = torch.randint(0,1,(1,length,2)).float() 
            seq = torch.cat([seq,aux],dim=2).requires_grad_(True)
            return seq
    
        def dense_decode(self,seq):
            ''' A method to convert a dense sequence to a readable nucleotide sequence'''
            as_list = seq.cpu().numpy().ravel().tolist()
            mapping = {0:'A',1:'C',2:'G',3:'T',4}#:'<score>',5:'<pad>'}
            nucs = [mapping[x] for x in as_list]
            return ''.join(nucs)
    
        def oracles(self):
            ''' A method to evaluate a sequence'''
            return [self.fwd]
    
        def seed_sequence(self):
            '''Generate a random sequence of a given length'''
            return torch.randint(0,4,(1,500)) 

if __name__ == "__main__":

    args = parse_args()
    device = "cpu" if args.device < 0 else f"cuda:{args.device}"
    module = SalukiDegradation.load_from_checkpoint(args.checkpoint,
                                                         map_location=device)
    
    # extact PyTorch model and wrap 
    half_life = setup_model_from_lightning(module,device)
    fwd = partial(half_life,seq_len=[500])

    def human_half_life(model,*args,**kwargs): 
        # lower confidence bound on the property of interest for EvidentialRegression
        dist =  model(*args,**kwargs)
        print(f'dist {dist.shape}')
        return dist[:,0]

    if half_life.evidential:
        lcb = partial(lower_confidence_bound,fwd)
        designer = StabilityDesigner(lcb)
    else:
        fwd = partial(human_half_life,fwd)
        designer = StabilityDesigner(fwd)

    # all PR and MCMC trials
    run_all_trials(designer,args)

