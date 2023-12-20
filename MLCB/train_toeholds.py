import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from models.toeholds.data import *
from models.toeholds.model import *
import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--max_epochs',type=int,default=100)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--weight_decay',type=float,default=1e-3)
    parser.add_argument('--decile_limit',type=int,default=None)
    parser.add_argument('--n_layers',type=int,default=2)
    parser.add_argument('--model_dim',type=int,default=64)
    parser.add_argument('--embed_dim',type=int,default=64)
    parser.add_argument('--downsample',action='store_true')
    parser.add_argument('--evidential',action='store_true')
    parser.add_argument('--dropout',type=float,default=0.3)
    parser.add_argument('--stop_metric',type=str,default='val_r2')
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--gradient_clip_val',type=float,default=0.5)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    
    df = parse_toehold('data/toehold_valeri_etal/downsampled_pruned_old_data.txt')
    test_set,valid_set,train_set = make_dataset_splits(df,42,decile_limit=args.decile_limit)
    train_loader = dataloader_from_dataset(train_set,args.batch_size)
    val_loader = dataloader_from_dataset(valid_set,args.batch_size) 
    test_loader = dataloader_from_dataset(test_set,args.batch_size)
    
    wandb_logger = pl.loggers.WandbLogger(project="toehold_onoff")
    wandb_name = wandb_logger.experiment.name
  
    max_epochs = args.max_epochs 
    patience = max_epochs // 10

    stop_metric = args.stop_metric
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
    module = ToeholdRegressor(n_layers=args.n_layers,
                              model_dim=args.model_dim,
                              embed_dim=args.embed_dim,
                              steps_per_epoch=steps_per_epoch,
                              max_epochs=max_epochs,
                              downsample=args.downsample,
                              evidential=args.evidential,
                              dropout=args.dropout,
                              learning_rate=args.lr)

    trainer = pl.Trainer(max_epochs=max_epochs,devices=1,
                         accelerator="gpu",logger=wandb_logger,
                         callbacks=[checkpoint_callback,early_stopping],
                         gradient_clip_val=args.gradient_clip_val)
    
    trainer.fit(module,train_loader,val_loader)
    print(f'Best model: {checkpoint_callback.best_model_path}')

    #trainer.test(module,test_loader,ckpt_path="best")