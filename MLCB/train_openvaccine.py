import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from models.openvaccine.data import dataloader_from_dataset, parse_json,make_dataset_splits
from models.openvaccine.model import OpenVaccineDegradation

if __name__ == "__main__":

    data = parse_json("data/open_vaccine_kaggle/train.json")
    train,val,test = make_dataset_splits(data,random_seed=42)
    batch_size = 32
    train_loader = dataloader_from_dataset(train,batch_size)
    val_loader = dataloader_from_dataset(val,batch_size) 
    test_loader = dataloader_from_dataset(test,batch_size)
    
    wandb_logger = pl.loggers.WandbLogger(project="openvaccine_deg")
    wandb_name = wandb_logger.experiment.name
    
    stop_metric = "val_MCRMSE"
    checkpoint_callback = ModelCheckpoint(
                        dirpath="models/openvaccine/checkpoints",
                        monitor=stop_metric,
                        save_top_k=5,
                        every_n_epochs=1,
                        filename=wandb_name+"-{epoch}-{val_MCRMSE:.4f}")

    early_stopping = EarlyStopping(stop_metric,patience=50)

    # train the model
    module = OpenVaccineDegradation() 
    trainer = pl.Trainer(max_epochs=1250,devices=1,
                         accelerator="gpu",logger=wandb_logger,
                         callbacks=[checkpoint_callback,early_stopping])
    trainer.fit(module,train_loader,val_loader)
    print(f'Best model: {checkpoint_callback.best_model_path}')
