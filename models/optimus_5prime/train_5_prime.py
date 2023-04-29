from torch import optim, nn
import lightning.pytorch as pl
from models.optimus_5prime.polysome import *
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

class OptimusFivePrime(torch.nn.Module):
    '''PyTorch implementation of https://github.com/pjsample/human_5utr_modeling'''
    
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv1d(in_channels=4,out_channels=120,kernel_size=8,padding='same')
        self.cnn2 = nn.Conv1d(in_channels=120,out_channels=120,kernel_size=8,padding='same')
        self.cnn3 = nn.Conv1d(in_channels=120,out_channels=120,kernel_size=8,padding='same')
        self.fc = nn.Linear(120 * 50,40)
        self.output = nn.Linear(40,1)
        
        self.stack1 = nn.Sequential(self.cnn1,
                                    nn.ReLU(),
                                    self.cnn2,
                                    nn.ReLU(),
                                    self.cnn3,
                                    nn.ReLU())
        
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
    
    def __init__(self):
        super().__init__()
        #self.save_hyperparameters()
        self.model = OptimusFivePrime()
        self.r2_score = torchmetrics.R2Score()

    def forward(self,x):
        return self.model(x)

    def setup_batch(self,batch):
        utr = torch.stack(batch['utr'],dim=0).to(self.device)
        utr = torch.nn.functional.one_hot(utr.squeeze(),num_classes=4).permute(0,2,1).float()
        target = torch.stack(batch['mrl'],dim=0).to(self.device).unsqueeze(1)
        return utr,target
    
    def training_step(self, batch, batch_idx):
        x,y = self.setup_batch(batch)
        y_pred = self.model(x)
        loss = F.mse_loss(y_pred,y)
        self.log("train_loss", loss,batch_size=x.shape[0])
        return loss
    
    def validation_step(self, batch, batch_idx):
        x,y = self.setup_batch(batch)
        y_pred = self.model(x)
        loss = F.mse_loss(y_pred,y)
        self.r2_score(y_pred,y)
        self.log("val_loss", loss,batch_size=x.shape[0])
        self.log("val_r2", self.r2_score,batch_size=x.shape[0])

    def test_step(self, batch, batch_idx):
        x,y = self.setup_batch(batch)
        y_pred = self.model(x)
        loss = F.mse_loss(y_pred,y)
        self.r2_score(y_pred,y)
        self.log("test_loss", loss,batch_size=x.shape[0])
        self.log("test_r2", self.r2_score,batch_size=x.shape[0])

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":

    test_df,train_df = parse_egfp_polysome("data/5-prime_sample_etal/GSM3130435_egfp_unmod_1.csv","data/5-prime_sample_etal/GSM3130436_egfp_unmod_2.csv")
    test_set,valid_set,train_set = make_dataset_splits(test_df,train_df,42)
    train_loader = dataloader_from_dataset(train_set,128)
    val_loader = dataloader_from_dataset(valid_set,128) 
    test_loader = dataloader_from_dataset(test_set,128)

    checkpoint_callback = ModelCheckpoint(
                        dirpath="models/optimus_5prime/checkpoints",
                        monitor="val_loss",
                        save_top_k=3,
                        every_n_epochs=1,
                        filename="{epoch}-{val_loss:.4f}")

    early_stopping = EarlyStopping("val_loss",patience=5)

    # train the model
    module = MeanRibosomeLoadModule() 
    wandb_logger = pl.loggers.WandbLogger(project="optimus_5prime")
    trainer = pl.Trainer(max_epochs=3,devices=1,
                         accelerator="gpu",logger=wandb_logger,
                         callbacks=[checkpoint_callback])
    trainer.fit(module,train_loader,val_loader)
    trainer.test(module,test_loader)