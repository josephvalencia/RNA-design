import torch
from torch import optim, nn
import lightning.pytorch as pl
import torch.nn.functional as F
import torchmetrics
from models.utils.evidential import EvidentialLoss, EvidentialRegressionOutputLayer
from models.utils.bytenet import ByteNetRNNRegression,ByteNetLayer

class HalfLife(ByteNetRNNRegression):

    def __init__(self,
                 n_layers=3,
                 model_dim=128,
                 embed_dim=128,
                 downsample=True,
                 pool_type='max',
                 rnn_type='gru',
                 evidential=False,
                 dropout=0.3,
                 max_dilation_factor=8):
        
        super().__init__(n_outputs=2,
                        embed_dim=embed_dim,
                        reduction='first',
                        model_dim=model_dim,
                        n_layers=n_layers,
                        downsample=downsample,
                        pool_type=pool_type,
                        rnn_type=rnn_type,
                        evidential=evidential,
                        dropout=dropout,
                        max_dilation_factor=max_dilation_factor)
        
        #self.embedding = nn.Embedding(6,embed_dim,padding_idx=5)
        self.in_cnn = nn.Conv1d(in_channels=8,
                                out_channels=embed_dim,
                                kernel_size=5,
                                padding='same')
        self.layernorm = nn.InstanceNorm1d(embed_dim,affine=True)
    
    def forward(self,x,seq_len):
        '''x : torch.Tensor of shape (batch_size,sequence_length)'''

        #x = self.embedding(x)
        x = self.in_cnn(x.permute(0,2,1))
        x = F.gelu(self.layernorm(x))
        x = x.permute(0,2,1)

        return super().forward(x,seq_len)

class SalukiDegradation(pl.LightningModule):

    def __init__(self,
                n_layers,
                model_dim,
                steps_per_epoch,
                max_epochs,
                embed_dim=64,
                downsample=True,
                pool_type='max',
                rnn_type='gru',
                evidential=False,
                dropout=0.3,
                max_dilation_factor=8,
                learning_rate=1e-3):
        
        super().__init__()
        
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = max_epochs
        self.evidential = evidential 
        self.learning_rate = learning_rate
        self.model = HalfLife(n_layers=n_layers,
                                        model_dim=model_dim,
                                        embed_dim=embed_dim,
                                        downsample=downsample,
                                        pool_type=pool_type,
                                        rnn_type=rnn_type,
                                        evidential=evidential,
                                        dropout=dropout,
                                        max_dilation_factor=max_dilation_factor)
        
        self.val_r2_score = torchmetrics.R2Score()
        self.train_r2_score = torchmetrics.R2Score()
        self.rmse_reactivity = torchmetrics.MeanSquaredError(squared=False)
        
        if self.evidential: 
            self.uq_spearman = torchmetrics.SpearmanCorrCoef()
            self.er_loss_train = EvidentialLoss(error_weight=0.05,reduction='mean')
            self.er_loss_val = EvidentialLoss(error_weight=0.05,reduction='mean')

        self.save_hyperparameters()

    def forward(self,seq,seq_len):
        return self.model(seq,seq_len)

    def setup_batch(self,batch):
        seq,seq_len,target,is_human = batch
        seq = seq.to(self.device)
        target = target.to(self.device)
        return seq,seq_len,target,is_human

    def shared_step(self, batch, batch_idx):
        seq,seq_len,y,is_human = self.setup_batch(batch)
        y_pred = self.model(seq,seq_len)
        
        # compute loss only over output head for correct species
        y_pred_masked = y_pred[:,0].squeeze(-1) if is_human else y_pred[:,1].squeeze(-1) 
        
        if self.evidential: 
            loss = self.er_loss_train(y_pred_masked,y) 
        else: 
            loss = F.mse_loss(y_pred_masked,y,reduction='mean')
        
        return y_pred_masked,y,loss

    def training_step(self, batch, batch_idx):
        y_pred,y,loss = self.shared_step(batch,batch_idx)

        B = y.shape[0] 
        if B > 1: 
            self.train_r2_score(y_pred,y)
            self.log("train_r2_score", self.train_r2_score,
                     batch_size=B,rank_zero_only=True,
                     on_step=False, on_epoch=True)
        self.log("train_loss", loss,batch_size=B,rank_zero_only=True)
        # don't bother to sync_dist for speed 
        return loss

    def validation_step(self, batch, batch_idx):
        
        y_pred,y,loss = self.shared_step(batch,batch_idx)
        B = y.shape[0]

        # sync_dist here for accuracy
        self.log("val_loss", loss,batch_size=B,
                 rank_zero_only=True,sync_dist=True)
        
        if B > 1: 
            self.val_r2_score(y_pred,y)
            self.log("val_r2_score", self.val_r2_score,
                        batch_size=B,rank_zero_only=True,sync_dist=True)
        if self.evidential: 
            self.uq_spearman(y_pred.epistemic_uncertainty,torch.abs(y_pred.mean-y))
            self.log("val_uq_spearman", self.uq_spearman,
                     batch_size=B,rank_zero_only=True,sync_dist=True)
            self.rmse_reactivity(y_pred.mean,y) 
        else: 
            self.rmse_reactivity(y_pred,y) 
        
        self.log(f"val_rmse_reactivity",self.rmse_reactivity,
                 batch_size=B,rank_zero_only=True,sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate,weight_decay=5e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                                  steps_per_epoch=self.steps_per_epoch,
                                                  epochs=self.trainer.max_epochs,
                                                  pct_start=0.1)
        schedule_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": schedule_config}