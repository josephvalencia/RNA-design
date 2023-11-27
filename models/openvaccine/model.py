import torch
from torch import optim, nn
import lightning.pytorch as pl
import torch.nn.functional as F
import torchmetrics
from models.utils.evidential import EvidentialLoss, EvidentialRegressionOutputLayer
from models.utils.bytenet import ByteNetRNNRegression

class SiteWiseDegradation(ByteNetRNNRegression):

    '''Embed seq,struct and loop then delegate to ByteNetRegression'''
    def __init__(self,
                 n_layers=3,
                 model_dim=128,
                 embed_dim=128,
                 downsample=True,
                 rnn_type='gru',
                 evidential=False,
                 dropout=0.2,
                 max_dilation_factor=1):
        
        super().__init__(n_outputs=3,
                        embed_dim=3*embed_dim,
                        reduction='none',
                        model_dim=model_dim,
                        n_layers=n_layers,
                        downsample=downsample,
                        pool_type='max',
                        rnn_type=rnn_type,
                        evidential=evidential,
                        dropout=dropout,
                        max_dilation_factor=max_dilation_factor)
        
        self.seq_embed = nn.Linear(5,embed_dim)
        self.struct_embed = nn.Linear(3,embed_dim)
        self.loop_embed = nn.Linear(7,embed_dim)

    def forward(self,seq,struct,loop,seq_lens):
        '''seq,struct,loop : torch.Tensors of shape (batch_size,sequence_length)'''
        
        seq_embed = self.seq_embed(seq)
        struct_embed = self.struct_embed(struct)
        loop_embed = self.loop_embed(loop)
        embed = torch.cat([seq_embed,struct_embed,loop_embed],dim=2)
        return super().forward(embed,seq_lens)

class OpenVaccineDegradation(pl.LightningModule):

    '''Predicts degradation rates and reactivity for each nucleotide given the sequence, structure, and loop type'''
    def __init__(self,
                n_layers=3,
                model_dim=128,
                embed_dim=32,
                downsample=True,
                rnn_type='gru',
                evidential=False,
                dropout=0.2,
                max_dilation_factor=1,
                learning_rate=1e-3):
        
        super().__init__() 
        
        self.evidential = evidential
        self.learning_rate = learning_rate
        self.model = SiteWiseDegradation(n_layers=n_layers,
                                        model_dim=model_dim,
                                        embed_dim=embed_dim,
                                        downsample=downsample,
                                        rnn_type=rnn_type,
                                        evidential=evidential,
                                        dropout=dropout,
                                        max_dilation_factor=max_dilation_factor)
        
        self.prop_to_index = {'reactivity': 0,'deg_Mg_pH10': 1,'deg_Mg_50C': 2}
        self.r2_score = torchmetrics.R2Score()
        self.rmse_reactivity = torchmetrics.MeanSquaredError(squared=False)
        self.rmse_deg_Mg_pH10 = torchmetrics.MeanSquaredError(squared=False)
        self.rmse_deg_Mg_50C = torchmetrics.MeanSquaredError(squared=False) 
        
        if self.evidential: 
            self.er_loss_val = EvidentialLoss(error_weight=0.05,reduction='mean')
            self.uq_spearman = torchmetrics.SpearmanCorrCoef()
            self.er_loss_train = EvidentialLoss(error_weight=0.05,reduction='mean')
        
        self.save_hyperparameters()
    
    def forward(self,seq,struct,loop,seq_lens):
        return self.model(seq,struct,loop,seq_lens)

    def rmse_error_weighted(self,y_pred,y_true,error,alpha=0.5,beta=5.0):
        '''Compute MSE loss with error weighting'''
        loss = F.mse_loss(y_pred,y_true,reduction='none')
        loss_wt = loss * (alpha + torch.exp(-beta*error))
        loss_wt_per_pos = torch.sqrt(torch.mean(loss_wt,dim=1))
        return torch.mean(loss_wt_per_pos) 

    def setup_batch(self,batch):
        seq = torch.stack(batch['seq'],dim=0).to(self.device)
        seq = torch.nn.functional.one_hot(seq.squeeze(),num_classes=5).float()
        struct = torch.stack(batch['struct'],dim=0).to(self.device)
        struct = torch.nn.functional.one_hot(struct.squeeze(),num_classes=3).float()
        loop = torch.stack(batch['loop'],dim=0).to(self.device)
        loop = torch.nn.functional.one_hot(loop.squeeze(),num_classes=7).float()
        seq_lens = batch['seq_len']
        target = torch.stack(batch['degradation'],dim=0).to(self.device)
        error = torch.stack(batch['degradation_error'],dim=0).to(self.device)
        return seq,struct,loop,seq_lens,target,error
    
    def shared_step(self,batch,batch_idx):
        seq,struct,loop,seq_lens,y,err = self.setup_batch(batch)
        y_pred = self.model(seq,struct,loop,seq_lens)
        n_positions = y.shape[1]
        y_pred_trunc = y_pred[:,:n_positions,]
        loss = self.rmse_error_weighted(y_pred_trunc,y,err) 
        #loss = F.mse_loss(y_pred_trunc,y,reduction='mean')
        #loss = self.er_loss_train(y_pred,y) 
        return y_pred_trunc,y,loss

    def training_step(self, batch, batch_idx):

        y_pred,y,loss = self.shared_step(batch,batch_idx) 
        self.log("train_loss", loss,batch_size=y_pred.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        
        y_pred,y,loss = self.shared_step(batch,batch_idx)
        flat_y = y.reshape(-1,3)
        flat_y_pred = y_pred.reshape(-1,3)

        B = flat_y.shape[0]
        # compute metrics for each property, as well as the mean across all properties
        metrics = [self.rmse_reactivity,self.rmse_deg_Mg_pH10,self.rmse_deg_Mg_50C]
        for metric,(prop,idx) in zip(metrics,self.prop_to_index.items()):
            metric(flat_y_pred[:,idx],flat_y[:,idx])
            self.log(f"val_rmse_{prop}",metric,batch_size=B)

        if self.evidential: 
            self.uq_spearman(y_pred.epistemic_uncertainty,torch.abs(y_pred.mean-y))
            self.log("val_uq_spearman", self.uq_spearman,batch_size=B)
        
        self.log("val_loss", loss,batch_size=y.shape[0])
    
    def on_validation_epoch_end(self) -> None:
        # log the mean of the metrics across all properties
        metrics = [self.rmse_reactivity,self.rmse_deg_Mg_pH10,self.rmse_deg_Mg_50C]
        metric_vals = [m.compute() for m in metrics]
        mean_metric = torch.mean(torch.stack(metric_vals))
        self.log("val_MCRMSE",mean_metric)
        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer