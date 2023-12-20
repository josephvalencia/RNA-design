import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchdata.datapipes as dp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

class PolysomeEGFP(dp.iter.IterDataPipe):

    def __init__(self,df):
        self.df = df

    def __iter__(self):
        for sample in self.df.to_dict('records'):
            mrl = sample['mrl'] if 'mrl' in sample else None
            seq_id = sample['seq_id'] if 'seq_id' in sample else 'seq_'+str(sample['Unnamed: 0'])
            example = {'utr' : list(sample['utr']),
                       'seq_id' : seq_id,
                       'mrl' : mrl}     
            yield example 

def make_dataset_splits(df_test,df_train,random_seed):

    scaler = StandardScaler()
    df_train['mrl'] = scaler.fit_transform(df_train['rl'].to_numpy().reshape(-1,1)).squeeze().tolist() 
    df_test['mrl'] = scaler.transform(df_test['rl'].to_numpy().reshape(-1,1)).squeeze().tolist()
    print(scaler.mean_,scaler.var_) 
    print(df_train[['rl','mrl']].describe())
    print(df_train[['rl','mrl']].quantile([0.1*i for i in range(11)]))
    high = df_train[df_train['mrl'] > 1.3]
    print(f'high: {len(high)}/{len(df_train)} = {len(high)/len(df_train):.3f}') 
    sns.jointplot(data=df_train,x='mrl',y='total_reads',kind='hist')
    plt.tight_layout()
    plt.savefig('plots/hist_mrl.png')
    plt.close() 
    splits = train_test_split(df_train,test_size=0.1,random_state=random_seed)
    df_train,df_valid = splits[0],splits[1]
    print(df_test[['rl','mrl']].quantile([0.1*i for i in range(11)]))
    return PolysomeEGFP(df_test),PolysomeEGFP(df_valid),PolysomeEGFP(df_train)

def apply_transform(x):
    '''Transforms a string of nucleotides into a list of integers'''

    seq_id = x['seq_id']
    mapping = {'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3}
    utr = [mapping[x] for x in x['utr']]
    mrl = torch.tensor(x['mrl'],dtype=torch.float32) if 'mrl' in x else None
    return torch.tensor(utr,dtype=torch.int64),mrl,seq_id

def dataloader_from_dataset(dataset,batch_size):
    '''Use datapipes to make dataloaders and numericalize utr seq'''

    dataset = dataset.map(apply_transform)
    dataset = dataset.batch(batch_size)
    dataset = dataset.rows2columnar(['utr','mrl','seq_id'])
    dataloader = DataLoader(dataset,batch_size=None,shuffle=True)
    return dataloader

def parse_egfp_polysome(file1,decile_limit=None):

    df = pd.read_csv(file1)
    test_n = 20000
    total_n = 280000

    df = df.sort_values(by='total_reads',ascending=False)
    df = df[df['total_reads'] > 0.0] 
    df_test = df[:test_n]
    df_train = df[test_n:total_n]
    
    if decile_limit is not None:
        # add the high ribosome load examples to the training set
        df_train['rl_bin'] = pd.qcut(df_train['rl'],q=10,labels=False)
        high = df_train[df_train['rl_bin'] >= decile_limit -1]
        df_train = df_train[df_train['rl_bin'] < decile_limit]
        df_test = pd.concat([df_test,high])
        print(df_train['rl_bin'].value_counts())
    return df_test.sample(frac=1.0), df_train.sample(frac=1.0)

