import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from models.optimus_5prime.data import parse_egfp_polysome,make_dataset_splits, dataloader_from_dataset 
from models.optimus_5prime.model import MeanRibosomeLoadModule
import sys

if __name__ == "__main__":

    devices = [int(sys.argv[1])] if len(sys.argv) > 1 else 'auto'
    test_df,train_df = parse_egfp_polysome("data/5-prime_sample_etal/GSM3130435_egfp_unmod_1.csv",
            "data/5-prime_sample_etal/GSM3130436_egfp_unmod_2.csv",decile_limit=9)
    test_set,valid_set,train_set = make_dataset_splits(test_df,train_df,42)
    batch_size = 128
    train_loader = dataloader_from_dataset(train_set,batch_size)
    val_loader = dataloader_from_dataset(valid_set,batch_size) 
    test_loader = dataloader_from_dataset(test_set,batch_size)

    wandb_logger = pl.loggers.WandbLogger(project="optimus_5prime")
    wandb_name = wandb_logger.experiment.name
    print(wandb_name)
    
    stop_metric = "val_loss"
    checkpoint_callback = ModelCheckpoint(
                        dirpath="models/optimus_5prime/checkpoints",
                        monitor=stop_metric,
                        save_top_k=5,
                        every_n_epochs=1,
                        filename=wandb_name+"-{epoch}-{val_loss:.4f}")

    early_stopping = EarlyStopping(stop_metric,patience=5)
    
    # train the model
    module = MeanRibosomeLoadModule(evidential=False) 
    trainer = pl.Trainer(max_epochs=50,devices=1,
                         accelerator="gpu",logger=wandb_logger,
                         callbacks=[checkpoint_callback,early_stopping],
                         gradient_clip_val=0.5)
    trainer.fit(module,train_loader,val_loader)
    print(f'Best model: {checkpoint_callback.best_model_path}')
    #trainer.test(module,test_loader,ckpt_path="best")
