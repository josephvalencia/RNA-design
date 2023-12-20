import torch
from torch import optim, nn
import lightning.pytorch as pl
import torch.nn.functional as F
import torchmetrics
from models.utils.evidential import EvidentialLoss, EvidentialRegressionOutputLayer
from models.utils.bytenet import ByteNetRNNRegression,ByteNetLayer
from collections import defaultdict

class HalfLife(ByteNetRNNRegression):

    def __init__(self,
                 vocab_size=8,
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
        self.in_cnn = nn.Conv1d(in_channels=vocab_size,
                                out_channels=embed_dim,
                                kernel_size=5,
                                padding='same')
        #self.layernorm = nn.InstanceNorm1d(embed_dim,affine=True)
        self.layernorm = nn.LayerNorm(embed_dim,elementwise_affine=True)

    def reshaped_layernorm(self,x):
        x = x.permute(0,2,1)
        x = self.layernorm(x)
        return x.permute(0,2,1)

    def forward(self,x,seq_len):
        '''x : torch.Tensor of shape (batch_size,sequence_length)'''

        #x = self.embedding(x)
        x = self.in_cnn(x.permute(0,2,1))
        x = self.reshaped_layernorm(x)
        x = F.gelu(x)
        x = x.permute(0,2,1)

        return super().forward(x,seq_len)

class SalukiDegradation(pl.LightningModule):

    def __init__(self,
                n_layers,
                model_dim,
                steps_per_epoch,
                max_epochs,
                include_aux=True,
                embed_dim=64,
                downsample=True,
                pool_type='max',
                rnn_type='gru',
                evidential=False,
                dropout=0.3,
                max_dilation_factor=8,
                learning_rate=1e-3,
                weight_decay=5e-3):
        
        super().__init__()
        
        self.include_aux = include_aux
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = max_epochs
        self.evidential = evidential 
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        vocab_size = 6 if include_aux else 4
        self.model = HalfLife(vocab_size=vocab_size,
                                n_layers=n_layers,
                                model_dim=model_dim,
                                embed_dim=embed_dim,
                                downsample=downsample,
                                pool_type=pool_type,
                                rnn_type=rnn_type,
                                evidential=evidential,
                                dropout=dropout,
                                max_dilation_factor=max_dilation_factor)
        
        self.setup_torchmetrics()

        if self.evidential: 
            self.uq_spearman = torchmetrics.SpearmanCorrCoef()
            self.er_loss_train = EvidentialLoss(error_weight=0.05,reduction='mean')
            self.er_loss_val = EvidentialLoss(error_weight=0.05,reduction='mean')

        self.save_hyperparameters()

    def setup_torchmetrics(self):

        species = ['mouse','human']
        datasets = ['train','val']
        
        for d in datasets:
            species_dict = torch.nn.ModuleDict()
            for s in species:
                metric_dict = torch.nn.ModuleDict({'R2' : torchmetrics.R2Score(),
                                'pearson' : torchmetrics.PearsonCorrCoef()})
                species_dict[s] = metric_dict
            setattr(self,f'{d}_metrics',species_dict)

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
        species = 'human' if is_human else 'mouse'
        # compute loss only over output head for correct species
        y_pred_masked = y_pred[:,0].squeeze(-1) if is_human else y_pred[:,1].squeeze(-1) 
        
        if self.evidential: 
            loss = self.er_loss_train(y_pred_masked,y) 
        else: 
            loss = F.mse_loss(y_pred_masked,y,reduction='mean')
        
        return y_pred_masked,y,loss,species

    def training_step(self, batch, batch_idx):
        y_pred,y,loss,species = self.shared_step(batch,batch_idx)
        B = y.shape[0] 
        
        # don't bother to sync_dist for speed 
        self.log("train_loss",loss,batch_size=B,rank_zero_only=True)
        if B > 1: 
            for m,tmetric in self.train_metrics[species].items():
                tmetric(y_pred,y)
                name = f'train_{species}_{m}'
                self.log(name,
                         tmetric,
                         batch_size=B,
                         rank_zero_only=True,
                         on_step=False,
                         on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        
        y_pred,y,loss,species = self.shared_step(batch,batch_idx)
        B = y.shape[0]

        # sync_dist here for accuracy
        self.log("val_loss", loss,batch_size=B,
                 rank_zero_only=True,sync_dist=True)
        
        if self.evidential: 
            self.uq_spearman(y_pred.epistemic_uncertainty,torch.abs(y_pred.mean-y))
            self.log("val_uq_spearman", self.uq_spearman,
                     batch_size=B,rank_zero_only=True,sync_dist=True)
            y_pred = y_pred.mean

        if B > 1: 
            for m,tmetric in self.val_metrics[species].items():
                tmetric(y_pred,y)
                name = f'val_{species}_{m}'
                self.log(name,
                         tmetric,
                         batch_size=B,
                         rank_zero_only=True,
                         on_step=False,
                         on_epoch=True,
                         sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate,weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                                  steps_per_epoch=self.steps_per_epoch,
                                                  epochs=self.trainer.max_epochs,
                                                  pct_start=0.02)
        schedule_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": schedule_config}