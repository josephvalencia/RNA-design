import torch
from torch import optim, nn
import lightning.pytorch as pl
import torchmetrics
import torch.nn.functional as F
from models.utils.evidential import EvidentialLoss, EvidentialRegressionOutputLayer, ERVirtualAdversarialLoss
from models.utils.bytenet import ByteNetRNNRegression

class MeanRibosomeLoad(ByteNetRNNRegression):

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
                        pool_type='max',
                        model_dim=model_dim,
                        n_layers=n_layers,
                        downsample=downsample,
                        rnn_type=rnn_type,
                        evidential=evidential,
                        dropout=dropout,
                        max_dilation_factor=max_dilation_factor)

        #self.embedding = nn.Linear(4,embed_dim)
        self.in_cnn = nn.Conv1d(in_channels=4,
                                out_channels=embed_dim,
                                kernel_size=5,
                                padding='same')
        self.layernorm = nn.InstanceNorm1d(embed_dim,affine=True)
    
    def forward(self,x):
        '''x : torch.Tensor of shape (batch_size,sequence_length)'''
        #x = self.embedding(x)
        x = self.in_cnn(x.permute(0,2,1))
        x = F.gelu(self.layernorm(x))
        x = x.permute(0,2,1)
        return super().forward(x)

class OptimusFivePrime(torch.nn.Module):
    '''PyTorch implementation of https://github.com/pjsample/human_5utr_modeling'''
    
    def __init__(self,hidden_size=120,kernel_size=8):
        super().__init__()
        self.cnn1 = nn.Conv1d(in_channels=4,out_channels=hidden_size,kernel_size=kernel_size,padding='same')
        self.cnn2 = nn.Conv1d(in_channels=hidden_size,out_channels=120,kernel_size=kernel_size,padding='same')
        self.cnn3 = nn.Conv1d(in_channels=hidden_size,out_channels=120,kernel_size=kernel_size,padding='same')
        self.fc = nn.Linear(hidden_size * 50,40)
        #self.output = nn.Linear(40,1)
        self.output = EvidentialRegressionOutputLayer(40)

        self.stack1 = nn.Sequential(self.cnn1,
                                    nn.ReLU(),
                                    self.cnn2,
                                    nn.Dropout(p=0.1),
                                    nn.ReLU(),
                                    self.cnn3,
                                    nn.ReLU(),
                                    nn.Dropout(p=0.1))
        
        self.stack2 = nn.Sequential(self.fc,
                                    nn.ReLU(),
                                    nn.Dropout(p=0.2),
                                    self.output) 

    def forward(self,x):
        z = self.stack1(x)
        z = z.reshape(z.shape[0],-1)
        y_prox = self.stack2(z)
        return y_prox

# define the LightningModule
class MeanRibosomeLoadModule(pl.LightningModule):
    
    def __init__(self,
                n_layers=3,
                model_dim=128,
                embed_dim=128,
                downsample=True,
                rnn_type='gru',
                evidential=True,
                dropout=0.2,
                max_dilation_factor=1,
                learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.evidential = evidential
        
        #self.model = OptimusFivePrime()
        self.model = MeanRibosomeLoad(n_layers=n_layers,
                                        model_dim=model_dim,
                                        embed_dim=embed_dim,
                                        downsample=downsample,
                                        rnn_type=rnn_type,
                                        evidential=evidential,
                                        dropout=dropout,
                                        max_dilation_factor=max_dilation_factor)
        
        self.r2_score = torchmetrics.R2Score()
        self.mse_train = nn.MSELoss()
        self.mse_val = nn.MSELoss()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        
        if self.evidential:
            self.uq_spearman = torchmetrics.SpearmanCorrCoef()
            self.er_loss_train = EvidentialLoss(error_weight=0.1,reduction='mean')
            self.er_loss_val = EvidentialLoss(error_weight=0.1,reduction='mean')
            self.adv_loss_train = ERVirtualAdversarialLoss(exact=True)
            self.adv_loss_val = ERVirtualAdversarialLoss(exact=True)
        
        self.save_hyperparameters()

    def forward(self,x):
        return self.model(x)

    def setup_batch(self,batch):
        utr = torch.stack(batch['utr'],dim=0).to(self.device)
        utr = torch.nn.functional.one_hot(utr.squeeze(),num_classes=4).float()#.permute(0,2,1).float()
        print(f'utr shape {utr.shape}')
        target = torch.stack(batch['mrl'],dim=0).to(self.device).unsqueeze(1)
        return utr,target
    
    def training_step(self, batch, batch_idx):
        x,y = self.setup_batch(batch)

        #LDS_loss = self.adv_loss_train(self.model,x)
        y_pred = self.model(x)
        loss = self.er_loss_train(y_pred,y) # + 0.1*LDS_loss 
        #loss = self.mse_train(y_pred,y)
        self.log("train_loss", loss,batch_size=x.shape[0])
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        x,y = self.setup_batch(batch)
        #LDS_loss = self.adv_loss_train(self.model,x)
        y_pred = self.model(x)
        loss = self.er_loss_val(y_pred,y) # + 0.1*LDS_loss 
        #loss = self.mse_val(y_pred,y)
        
        #self.r2_score(y_pred,y)
        #self.rmse(y_pred,y)
        
        self.r2_score(y_pred.mean,y)
        self.rmse(y_pred.mean,y)
        self.uq_spearman(y_pred.epistemic_uncertainty,torch.abs(y_pred.mean-y))
        
        B = x.shape[0] 
        self.log("val_rmse", self.rmse,batch_size=B)
        self.log("val_uq_spearman", self.uq_spearman,batch_size=B)
        self.log("val_loss", loss,batch_size=B)
        self.log("val_r2", self.r2_score,batch_size=B)

    def test_step(self, batch, batch_idx):
        x,y = self.setup_batch(batch)
        y_pred = self.model(x)
        loss = self.er_loss_val(y_pred,y)
        self.r2_score(y_pred.mean,y)
        #loss =  self.mse_val(y_pred,y)
        #self.r2_score(y_pred,y) 
        self.log("test_loss", loss,batch_size=x.shape[0])
        self.log("test_r2", self.r2_score,batch_size=x.shape[0])

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


