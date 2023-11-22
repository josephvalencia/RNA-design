import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from models.saluki.data import  dataloader_from_json
from models.saluki.model import SalukiDegradation
from lightning.pytorch.utilities import rank_zero_only

if __name__ == "__main__":

    data_dir = "data/saluki_agarwal_kelley/"
    batch_size = 64
    train_loader = dataloader_from_json(data_dir,'train',batch_size)
    val_loader = dataloader_from_json(data_dir,'valid',batch_size) 
    test_loader = dataloader_from_json(data_dir,'test',batch_size)
    
    wandb_logger = pl.loggers.WandbLogger(project="saluki_deg")
    wandb_name = wandb_logger.experiment.name if rank_zero_only.rank == 0 else 'dummy_name'
    
    stop_metric = "val_loss"
    checkpoint_callback = ModelCheckpoint(
                        dirpath="models/saluki/checkpoints",
                        monitor=stop_metric,
                        save_top_k=5,
                        every_n_epochs=1,
                        filename=wandb_name+"-{epoch}-{val_loss:.4f}")

    max_epochs = 100
    patience = max_epochs // 10
    early_stopping = EarlyStopping(stop_metric,patience=patience)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    steps_per_epoch = 0
    for batch in train_loader:
        steps_per_epoch += 1
    print(f'N_STEPS_PER_EPOCH: {steps_per_epoch}')
    
    # train the model
    module = SalukiDegradation(n_layers=4,
                               model_dim=64,
                               embed_dim=64,
                               steps_per_epoch=steps_per_epoch,
                               max_epochs=max_epochs)

    trainer = pl.Trainer(max_epochs=max_epochs,devices=[1],
                         accelerator="gpu",#strategy="ddp",
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback,early_stopping,lr_monitor],
                         gradient_clip_val=0.5,
                         precision='16-mixed')
                         #accumulate_grad_batches=4)
    trainer.fit(module,train_loader,val_loader)
    print(f'Best model: {checkpoint_callback.best_model_path}')
