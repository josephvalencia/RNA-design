import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from models.toeholds.data import *
from models.toeholds.model import *

if __name__ == "__main__":

    df = parse_toehold('data/toehold_valeri_etal/downsampled_pruned_old_data.txt')
    test_set,valid_set,train_set = make_dataset_splits(df,42)
    batch_size = 32 
    train_loader = dataloader_from_dataset(train_set,batch_size)
    val_loader = dataloader_from_dataset(valid_set,batch_size) 
    test_loader = dataloader_from_dataset(test_set,batch_size)
    
    wandb_logger = pl.loggers.WandbLogger(project="toehold_onoff")
    wandb_name = wandb_logger.experiment.name
  
    max_epochs = 100 
    patience = max_epochs // 10

    stop_metric = "val_r2"
    checkpoint_callback = ModelCheckpoint(
                        dirpath="models/toeholds/checkpoints",
                        monitor=stop_metric,
                        save_top_k=5,
                        every_n_epochs=1,
                        filename=wandb_name+"-{epoch}-{val_loss:.4f}_{val_r2:.4f}",
                        mode="max")

    steps_per_epoch = 0
    for batch in train_loader:
        steps_per_epoch += 1
    print(f'N_STEPS_PER_EPOCH: {steps_per_epoch}')
    
    early_stopping = EarlyStopping(stop_metric,patience=patience,mode="max")

    # train the model
    module = ToeholdRegressor(n_layers=2,
                              model_dim=64,
                              steps_per_epoch=steps_per_epoch,
                              max_epochs=max_epochs) 
    trainer = pl.Trainer(max_epochs=max_epochs,devices=1,
                         accelerator="gpu",logger=wandb_logger,
                         callbacks=[checkpoint_callback,early_stopping])
    
    trainer.fit(module,train_loader,val_loader)
    print(f'Best model: {checkpoint_callback.best_model_path}')

    #trainer.test(module,test_loader,ckpt_path="best")