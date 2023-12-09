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
            #counts = np.asarray([sample[f'{x}'] for x in range(14)])
            #ribosomes = np.asarray([sample[f'r{x}'] for x in range(14)])
            #reads = sample['total_reads'] 
            #example = {'utr' : list(sample['utr']), 'mrl' : sample['mrl'],'reads' : reads}     
            example = {'utr' : list(sample['utr']),'seq_id' : sample['seq_id']}     
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
    return torch.tensor(utr,dtype=torch.int64),seq_id

def dataloader_from_dataset(dataset,batch_size):
    '''Use datapipes to make dataloaders and numericalize utr seq'''

    dataset = dataset.map(apply_transform)
    dataset = dataset.batch(batch_size)
    #dataset = dataset.rows2columnar(['utr','mrl'])
    dataset = dataset.rows2columnar(['utr','seq_id'])
    dataloader = DataLoader(dataset,batch_size=None,shuffle=True)
    return dataloader

def parse_egfp_polysome(file1,file2,decile_limit=None):

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    new_cols = { x : y for x,y in zip(df2.columns,df1.columns) }
    df2 = df2.rename(columns=new_cols)
    #combined = pd.concat([df1,df2])
    combined = df1
    test_n = 20000
    total_n = 280000

    combined = combined.sort_values(by='total_reads',ascending=False)
    combined = combined[combined['total_reads'] > 0.0] 
    df_test = combined[:test_n]
    df_train = combined[test_n:total_n]
    
    if decile_limit is not None:
        # add the high ribosome load examples to the training set
        df_train['rl_bin'] = pd.qcut(df_train['rl'],q=10,labels=False)
        high = df_train[df_train['rl_bin'] > 8]
        df_train = df_train[df_train['rl_bin'] <= 8]
        df_test = pd.concat([df_test,high])
        print(df_train['rl_bin'].value_counts())
    return df_test.sample(frac=1.0), df_train.sample(frac=1.0)

