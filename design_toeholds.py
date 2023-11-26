import torch
from models.toeholds.model import ToeholdRegressor
from seqopt.oracle import NucleotideDesigner
from trials import parse_args, setup_model_from_lightning, lower_confidence_bound,run_all_trials,tune_langevin
from functools import partial
import random
import pandas as pd

from models.toeholds.data import *


class ToeholdDesigner(NucleotideDesigner):
        
        def __init__(self,forward_fn):
            super().__init__(num_classes=4,class_dim=2)
            self.fwd = forward_fn
            self.setup_data()

        def setup_data(self): 
            self.train_df = parse_toehold('data/toehold_valeri_etal/downsampled_pruned_old_data.txt')

        def onehot_encode(self,seq):
            '''Index tensor to one-hot encoding'''
            seq = torch.nn.functional.one_hot(seq,num_classes=4).float().requires_grad_(True)
            return seq
    
        def dense_decode(self,seq):
            ''' A method to convert a dense sequence to a readable nucleotide sequence'''
            as_list = seq.cpu().numpy().ravel().tolist()
            mapping = {0:'A',1:'C',2:'G',3:'T'}
            nucs = [mapping[x] for x in as_list]
            return ''.join(nucs)
    
        def oracles(self):
            ''' A method to evaluate a sequence'''
            return [self.fwd]
    
        def reverse_complement(self,base_index): 
            # ACGT = alphabet
            if base_index == 'A':
                return 'T'
            elif base_index == 'C':
                return 'G' 
            elif base_index == 'G':
                return 'C' 
            elif base_index == 'T':
                return 'A'
        
        def seed_sequence(self):
            '''Generate a random sequence'''
            mapping = {'A' : 0,'C' : 1,'G' : 2,'T':3}
            shine_dalgarno = [char for char in 'AACAGAGGAGA']
            start_codon = [char for char in 'ATG']
            template =  random.choices(['A','C','T','G'],k=30)
            first_stem = template[12:21]
            second_stem = template[24:30]
            first_stem_partner = [self.reverse_complement(x) for x in first_stem[::-1]]
            second_stem_partner = [self.reverse_complement(x) for x in second_stem[::-1]]
            a = ''.join(first_stem)
            b = ''.join(second_stem)
            a_prime = ''.join(first_stem_partner)
            b_prime = ''.join(second_stem_partner)
            #print(f'a: {a}, b: {b}, a\': {a_prime}, b\': {b_prime}')
            toehold = template+shine_dalgarno+second_stem_partner+start_codon+first_stem_partner
            print(len(toehold))
            #toehold = 'AAAAAAATTAAACATTGAAAAGGTGTCTAGAACAGAGGAGACTAGACATGTTTTCAATG'
            #toehold = self.train_df.iloc[random.randint(0,len(self.train_df)-1)]['switch_sequence'] 
            #return torch.randint(0,4,(1,59))
            return torch.tensor([mapping[x] for x in toehold]).unsqueeze(0)
        
if __name__ == "__main__":

    args = parse_args()
    device = "cpu" if args.device < 0 else f"cuda:{args.device}"
    module = ToeholdRegressor.load_from_checkpoint(args.checkpoint,map_location=device)
    
    # extact PyTorch model and wrap 
    toehold_regression = setup_model_from_lightning(module,device)

    def onoff_ratio(model,*args,**kwargs):
        preds = model(*args,**kwargs)
        onoff =  preds[:,0] - preds[:,1]
        return onoff 
    
    # wrap on,off output into on - off and lower confidence bound if needed
    if toehold_regression.evidential:
        lcb = partial(lower_confidence_bound,toehold_regression)
        onoff = partial(onoff_ratio,lcb)
        designer = ToeholdDesigner(onoff)
    else: 
        onoff = partial(onoff_ratio,toehold_regression)
        designer = ToeholdDesigner(onoff)

    # all PR and MCMC trials
    #tune_langevin(designer,args)
    run_all_trials(designer,args)

