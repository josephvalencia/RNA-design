import torch
from torch import optim, nn
import lightning.pytorch as pl
import torchmetrics
from models.utils.evidential import EvidentialLoss
from models.utils.bytenet import ByteNetRNNRegression

class ToeholdOnOff(ByteNetRNNRegression):
    
    def __init__(self,
                 n_layers=3,
                 model_dim=128,
                 embed_dim=128,
                 downsample=True,
                 rnn_type='gru',
                 evidential=False,
                 dropout=0.2,
                 max_dilation_factor=1):
        
        super().__init__(n_outputs=1,
                        embed_dim=embed_dim,
                        reduction='first',
                        model_dim=model_dim,
                        n_layers=n_layers,
                        downsample=downsample,
                        rnn_type=rnn_type,
                        evidential=evidential,
                        dropout=dropout,
                        max_dilation_factor=max_dilation_factor)
        self.embedding = nn.Linear(5,embed_dim)
    
    def forward(self,x):
        '''x : torch.Tensor of shape (batch_size,sequence_length)'''

        x = self.embedding(x)
        return super().forward(x)

class ToeholdRegressor(pl.LightningModule):

    def __init__(self,
                n_layers=3,
                model_dim=128,
                embed_dim=128,
                downsample=True,
                rnn_type='gru',
                evidential=False,
                dropout=0.2,
                max_dilation_factor=1,
                learning_rate=1e-3):

        super().__init__() 
        self.model = ToeholdOnOff(n_layers=n_layers,
                                        model_dim=model_dim,
                                        embed_dim=embed_dim,
                                        downsample=downsample,
                                        rnn_type=rnn_type,
                                        evidential=evidential,
                                        dropout=dropout,
                                        max_dilation_factor=max_dilation_factor)
        self.mse_train = nn.MSELoss()
        self.mse_val = nn.MSELoss()
        self.r2_score = torchmetrics.R2Score()
        self.uq_spearman = torchmetrics.SpearmanCorrCoef()
        self.rmse = torchmetrics.MeanSquaredError(squared=False) 
        self.mae = torchmetrics.MeanAbsoluteError()
        self.er_loss_train = EvidentialLoss(error_weight=0.05,reduction='mean')
        self.er_loss_val = EvidentialLoss(error_weight=0.05,reduction='mean')
        self.metric = torchmetrics.MeanSquaredError()
        self.learning_rate = 1e-4

    def forward(self,x):
        return self.model(x)

    def setup_batch(self,batch):
        seq = torch.stack(batch['seq'],dim=0).to(self.device)
        seq = torch.nn.functional.one_hot(seq.squeeze(),num_classes=5).float()
        target = torch.stack(batch['on'],dim=0).to(self.device).unsqueeze(1)
        return seq,target
    
    def training_step(self, batch, batch_idx):
        
        x,y = self.setup_batch(batch)
        y_pred = self.model(x)
        loss = self.mse_train(y_pred,y) 
        #loss = self.er_loss_train(y_pred,y) 
        self.log("train_loss", loss,batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = self.setup_batch(batch)
        y_pred = self.model(x)
        
        loss = self.mse_val(y_pred,y)
        self.r2_score(y_pred,y)
        self.rmse(y_pred,y)
        self.mae(y_pred,y)
        
        #loss = self.er_loss_val(y_pred,y) 
        #self.r2_score(y_pred.mean,y)
        #self.rmse(y_pred.mean,y)
        #self.mae(y_pred.mean,y)
        #self.uq_spearman(y_pred.epistemic_uncertainty,torch.abs(y_pred.mean-y))
        
        B = x.shape[0]
        self.log("val_rmse", self.rmse,batch_size=B)
        self.log("val_mae", self.mae,batch_size=B)
        #self.log("val_uq_spearman", self.uq_spearman,batch_size=B)
        self.log("val_loss", loss,batch_size=B)
        self.log("val_r2", self.r2_score,batch_size=B)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

def stem_base_pairing(pwm): 
    
    '''Adapted from https://github.com/midas-wyss/engineered-riboregulator-ML/blob/master/example_storm_optimization.ipynb'''
    # build loss function
    # ensure biological constraints are satisfied per sequence
    # ensure that location of 1s in switch region matches reverse complement of stem
    
    def reverse_complement(base_index): 
        # ACGT = alphabet
        if base_index == 0: return 3
        elif base_index == 1: return 2 
        elif base_index == 2: return 1 
        elif base_index == 3: return 0
    
    # reverse complement is reverse over axis of one-hot encoded nt 
    nt_reversed = K.reverse(pwm, axes=2)
    
    #stem1_score = 6 - torch.sum(pwm[:, 24, :, 0]*nt_reversed[:, 41,:, 0] + pwm[:, 25, :, 0]*nt_reversed[:, 42, :, 0]+ pwm[:,26, :, 0]*nt_reversed[:, 43, :, 0] + pwm[:, 27, :, 0]*nt_reversed[:, 44, :, 0] + pwm[:, 28, :, 0]*nt_reversed[:, 45, :, 0]+ pwm[:, 29, :, 0]*nt_reversed[:, 46, :, 0])
    #stem2_score = 9 - torch.sum(pwm[:, 12, :, 0]*nt_reversed[:, 50, :, 0] + pwm[:, 13, :, 0]*nt_reversed[:, 51, :, 0]+ pwm[:, 14, :, 0]*nt_reversed[:, 52, :, 0]+ pwm[:, 15, :, 0]*nt_reversed[:, 53, :, 0] + pwm[:, 16, :, 0]*nt_reversed[:, 54, :, 0] + pwm[:, 17, :, 0]*nt_reversed[:,55, :, 0]+ pwm[:, 18,:, 0]*nt_reversed[:, 56, :, 0] + pwm[:, 19, :, 0]*nt_reversed[:,57, :, 0] + pwm[:, 20, :, 0]*nt_reversed[:, 58, :, 0])
    
    # count up from 24-29 on pwm, count up from 41-46 on nt_reversed
    stem1_score = 6 - torch.dot(pwm[:,24:30,:,0],nt_reversed[:,41:47,:,0]).sum() 
    # count up from 12-20 on pwm, count up from 50-58 on nt_reversed 
    stem2_score = 9 - torch.dot(pwm[:,12:21,:,0],nt_reversed[:,50:59,:,0]).sum() 
    
    return 10*stem1_score + 10*stem2_score

    