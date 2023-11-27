import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.evidential import EvidentialRegressionOutputLayer

class ByteNetLayer(torch.nn.Module):
    '''PyTorch implementation of ByteNet from https://arxiv.org/abs/1610.10099
      as modified in https://openreview.net/pdf?id=3i7WPak2sCx'''
    
    def __init__(self,dim=128,
                 dilation_rate=1,
                 downsample=False,
                 dropout=0.2,
                 inner_kernel_size=5):
        
        super().__init__()

        lower_dim = dim // 2 if downsample else dim
        #self.layernorm1 = nn.InstanceNorm1d(dim,affine=True)
        #self.layernorm2 = nn.InstanceNorm1d(lower_dim,affine=True)
        #self.layernorm3 = nn.InstanceNorm1d(lower_dim,affine=True)
        self.layernorm1 = nn.LayerNorm(dim,elementwise_affine=True)
        self.layernorm2 = nn.LayerNorm(lower_dim,elementwise_affine=True)
        self.layernorm3 = nn.LayerNorm(lower_dim,elementwise_affine=True)

        self.dropout1 = nn.Dropout(dropout)

        self.cnn1 = nn.Conv1d(in_channels=dim,
                              out_channels=lower_dim,
                              kernel_size=1)

        self.cnn2 = nn.Conv1d(in_channels=lower_dim,
                              out_channels=lower_dim,
                              kernel_size=inner_kernel_size,
                              dilation=dilation_rate,
                              padding='same')
        
        self.cnn3 = nn.Conv1d(in_channels=lower_dim,
                              out_channels=dim,
                              kernel_size=1)
        
    def reshaped_layernorm(self,layernorm,x):
        x = x.permute(0,2,1)
        x = layernorm(x)
        return x.permute(0,2,1)

    def forward(self,x):
        '''x : torch.Tensor of shape (batch_size,embedding_size,sequence_length)'''

        residual = x  
        #x = self.layernorm1(x)
        x = self.reshaped_layernorm(self.layernorm1,x)
        x = F.gelu(x)
        x = self.cnn1(x)
        #x = self.layernorm2(x)
        x = self.reshaped_layernorm(self.layernorm2,x)
        x = F.gelu(x)
        x = self.cnn2(x)
        #x = self.layernorm3(x)
        x = self.reshaped_layernorm(self.layernorm3,x)
        x = F.gelu(x)
        x = self.cnn3(x)
        return self.dropout1(x)+residual

class ByteNetRNNRegression(torch.nn.Module):
    
    def __init__(self,n_outputs,embed_dim,reduction,
                 model_dim=128,n_layers=10,downsample=True,
                 pool_type='none',rnn_type='gru',evidential=False,
                 dropout=0.2,max_dilation_factor=0):
        
        super().__init__()

        if reduction not in ['mean','first','none']:
            raise ValueError('reduction must be one of "mean","first","none"')
        self.reduction = reduction
        self.pool_type = pool_type
        self.evidential = evidential 

        # option for exponentially increasing dilation rate up to 2^max_dilation_factor
        dilation_factor = lambda l: 2**(l % max_dilation_factor) if max_dilation_factor > 0 else 1
        self.cnn_layers = nn.ModuleList([ByteNetLayer(model_dim,
                                                    dilation_rate=dilation_factor(i),
                                                    downsample=downsample,
                                                    dropout=dropout) for i in range(n_layers)])

        RNN = nn.GRU if rnn_type == 'gru' else nn.LSTM
        self.in_rnn = RNN(embed_dim,model_dim // 2,
                            batch_first=True,bidirectional=True) 
        self.out_rnn = RNN(model_dim,model_dim // 2,
                            batch_first=True,bidirectional=True)

        # option to pool in sequence dimension
        if self.pool_type == 'max':
            self.pool =  nn.MaxPool1d(2)
        elif self.pool_type == 'avg':
            self.pool = nn.AvgPool1d(2)  

        if self.evidential:
            self.output = EvidentialRegressionOutputLayer(model_dim)
        else:
            self.output = nn.Linear(model_dim,n_outputs)


    def apply_rnn(self,layer,x,seq_lens=None):
        '''maybe pack/unpack batch so that RNN only operates on non-padded positions'''
        # x : torch.Tensor of shape (batch_size,sequence_length,embedding_size)
        # seq_lens : list of length batch_size
        
        if seq_lens is not None: 
            packed = nn.utils.rnn.pack_padded_sequence(x,seq_lens,
                                                    batch_first=True,
                                                    enforce_sorted=False) 
            output, _ = layer(packed)
            unpacked, _ = nn.utils.rnn.pad_packed_sequence(output,batch_first=True)
            return unpacked
        else:
            return layer(x)[0] 
        
    def forward(self,seq_embed,seq_lens=None):
        '''x : torch.Tensor of shape (batch_size,sequence_length)'''
        
        # RNNs expect input of shape (batch_size,sequence_length,embedding_size)
        x = self.apply_rnn(self.in_rnn,seq_embed,seq_lens)
        
        # 1D ByteNet layers expect input of shape (batch_size,embedding_size,sequence_length) 
        x = x.permute(0,2,1)
        for layer in self.cnn_layers:
            x = layer(x)
            if self.pool_type != 'none':
                x = self.pool(x)
        x = x.permute(0,2,1)
        
        # seq_lens are no longer valid after pooling 
        lens = None if self.pool_type != 'none' else seq_lens
        x = self.apply_rnn(self.out_rnn,x,lens)

        # take summary of sequence to match target shape 
        if self.reduction == 'mean':
            summary = torch.mean(x,dim=1)
        elif self.reduction == 'first':
            summary = x[:,0,:]
        # no reduction
        else:
            summary = x
        
        # final regression layer 
        return self.output(summary)
