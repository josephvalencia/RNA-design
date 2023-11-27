import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchdata.datapipes as dp
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

class ToeholdData(dp.iter.IterDataPipe):

    def __init__(self,df):
        self.df = df

    def __iter__(self):
        for sample in self.df.to_dict('records'):
            example = {'switch_sequence' : list(sample['switch_sequence']),
                        'trigger_sequence' : list(sample['trigger_sequence']),
                        'on' : sample['on_value'],
                        'off' : sample['off_value'],
                        'onoff' : sample['onoff_value']}     
            yield example 

def make_dataset_splits(df,random_seed):

    df_train,df_test = train_test_split(df,test_size=0.1,random_state=random_seed) 
    sns.regplot(data=df_train,x='off_value',y='on_value')
    plt.tight_layout()
    plt.savefig('plots/hist_on_off.png')
    plt.close()
    print(df_train['onoff_value'].describe())
    print(df_train['onoff_value'].quantile([i*0.1 for i in range(11)]))
    splits = train_test_split(df_train,test_size=0.1,random_state=random_seed)
    df_train,df_valid = splits[0],splits[1]
    return ToeholdData(df_test),ToeholdData(df_valid),ToeholdData(df_train)

def apply_transform(x):
    '''Transforms a string of nucleotides into a list of integers'''

    mapping = {'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3,'<reg>' : 4}
    #seq = [mapping[x] for x in x['switch_sequence']]
    seq = [mapping[x] for x in x['trigger_sequence']]
    seq = torch.tensor(seq,dtype=torch.int64)
    on = torch.tensor(x['on'],dtype=torch.float32)
    off = torch.tensor(x['off'],dtype=torch.float32)
    onoff = torch.tensor(x['onoff'],dtype=torch.float32) 
    return seq,on,off,onoff

def dataloader_from_dataset(dataset,batch_size):
    '''Use datapipes to make dataloaders and numericalize utr seq'''

    dataset = dataset.map(apply_transform)
    dataset = dataset.batch(batch_size)
    dataset = dataset.rows2columnar(['seq','on','off','onoff'])
    dataloader = DataLoader(dataset,batch_size=None,shuffle=True)
    return dataloader

def parse_toehold(filename):

    df = pd.read_csv(filename)

    # downsample to 1000 bins for both ON and OFF and keep the union
    df['on_binned'] = pd.cut(df['on_value'],bins=1000)
    df['off_binned'] = pd.cut(df['off_value'],bins=1000)
    on_mean_count = int(df['on_binned'].value_counts().mean())
    off_mean_count = int(df['off_binned'].value_counts().mean()) 
    final_set = set()
    for bin,subdf in df.groupby('on_binned'):
        n = min(len(subdf),on_mean_count)
        final_set.update(subdf.sample(n=n).index) 
    for bin,subdf in df.groupby('off_binned'):
        n = min(len(subdf),off_mean_count) 
        final_set.update(subdf.sample(n=n).index)

    df = df.loc[list(final_set)]
    return df
