import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchdata.datapipes as dp
from sklearn.model_selection import train_test_split

class DegradationLoader(dp.iter.IterDataPipe):

    def __init__(self,data):
        self.data = data

    def __iter__(self):
        for sample in self.data.to_dict(orient='records'):
            yield sample

def make_dataset_splits(data,random_seed=65):
    '''Split the dataset into train, val, and test sets'''
    train,test = train_test_split(data,test_size=0.1,random_state=random_seed)
    train,val = train_test_split(train,test_size=0.1,random_state=random_seed)
    return DegradationLoader(test),DegradationLoader(val),DegradationLoader(train)

def apply_transform(x):
    '''Transforms a string of nucleotides into a list of integers'''

    vocab = {'A' : 0, 
            'C' : 1,
            'G' : 2,
            'U' : 3}
    
    bp_vocab = {'(' : 0,
                ')' : 1,
                '.' : 2}
    
    bprna_vocab = {'S' : 0,
                   'M' : 1,
                   'I' : 2,
                   'B' : 3,
                   'H' : 4,
                   'E' : 5,
                   'X' : 6}

    seq = [vocab[x] for x in x['sequence']]
    seq = torch.tensor(seq,dtype=torch.int64)

    seq_len = x['seq_length']

    struct = [bp_vocab[x] for x in x['structure']]
    struct = torch.tensor(struct,dtype=torch.int64)

    loop = [bprna_vocab[x] for x in x['predicted_loop_type']]
    loop = torch.tensor(loop,dtype=torch.int64)

    reactivity = torch.tensor(x['reactivity'],dtype=torch.float32)
    deg_Mg_pH10 = torch.tensor(x['deg_Mg_pH10'],dtype=torch.float32)
    deg_Mg_50C = torch.tensor(x['deg_Mg_50C'],dtype=torch.float32) 
    degradation = torch.stack([reactivity,deg_Mg_pH10,deg_Mg_50C],dim=1)

    react_error = torch.tensor(x['reactivity_error'],dtype=torch.float32)
    deg_Mg_pH10_error = torch.tensor(x['deg_error_Mg_pH10'],dtype=torch.float32)
    deg_Mg_50C_error = torch.tensor(x['deg_error_Mg_50C'],dtype=torch.float32)
    degradation_error = torch.stack([react_error,deg_Mg_pH10_error,deg_Mg_50C_error],dim=1)
    return seq,struct,loop,seq_len,degradation,degradation_error

def dataloader_from_dataset(dataset,batch_size):
    '''Use datapipes to make dataloaders and numericalize utr seq'''

    dataset = dataset.map(apply_transform)
    dataset = dataset.batch(batch_size)
    dataset = dataset.rows2columnar(['seq','struct','loop','seq_len','degradation','degradation_error'])
    dataloader = DataLoader(dataset,batch_size=None,shuffle=True)
    return dataloader

def parse_json(filename):
    return pd.read_json(filename,lines=True)