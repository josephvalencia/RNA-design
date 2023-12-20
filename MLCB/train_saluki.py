import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from models.saluki.data import  dataloader_from_json
from models.saluki.model import SalukiDegradation
from lightning.pytorch.utilities import rank_zero_only
import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--max_epochs',type=int,default=100)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--weight_decay',type=float,default=1e-3)
    parser.add_argument('--decile_limit',type=int,default=None)
    parser.add_argument('--n_layers',type=int,default=3)
    parser.add_argument('--model_dim',type=int,default=64)
    parser.add_argument('--embed_dim',type=int,default=64)
    parser.add_argument('--downsample',action='store_true')
    parser.add_argument('--evidential',action='store_true')
    parser.add_argument('--no_aux',action='store_true')
    parser.add_argument('--dropout',type=float,default=0.3)
    parser.add_argument('--stop_metric',type=str,default='val_loss')
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--gradient_clip_val',type=float,default=0.5)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    data_dir = "data/saluki_agarwal_kelley/"
    include_aux = not args.no_aux 
    train_loader = dataloader_from_json(data_dir,'train',
                                        include_aux,args.batch_size,
                                        decile_limit=args.decile_limit)
    val_loader = dataloader_from_json(data_dir,'valid',
                                      include_aux,args.batch_size,
                                      decile_limit=args.decile_limit) 
    test_loader = dataloader_from_json(data_dir,'test',
                                       include_aux,args.batch_size)
    
    wandb_logger = pl.loggers.WandbLogger(project="saluki_stability",
                                          save_dir="wandb/",)
    wandb_name = wandb_logger.experiment.name if rank_zero_only.rank == 0 else 'dummy_name'
    
    checkpoint_callback = ModelCheckpoint(
                        dirpath="models/saluki/checkpoints",
                        monitor=args.stop_metric,
                        save_top_k=5,
                        every_n_epochs=1,
                        filename=wandb_name+"-{epoch}-{val_loss:.4f}")
    
    max_epochs = args.max_epochs
    patience = max_epochs // 10
    early_stopping = EarlyStopping(args.stop_metric,patience=patience)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    steps_per_epoch = 0
    for batch in train_loader:
        steps_per_epoch += 1
    print(f'N_STEPS_PER_EPOCH: {steps_per_epoch}')
    # train the model
    module = SalukiDegradation(n_layers=args.n_layers,
                               model_dim=args.model_dim,
                               embed_dim=args.embed_dim,
                               downsample=args.downsample,
                               include_aux=include_aux,
                               dropout=args.dropout,
                               steps_per_epoch=steps_per_epoch,
                               max_epochs=max_epochs,
                               weight_decay=args.weight_decay)
    
    devices = [args.device] if args.device >= 0 else 'cpu'
    accelerator = "gpu" if args.device >= 0 else "cpu"
    trainer = pl.Trainer(max_epochs=max_epochs,
                         devices=devices,
                         accelerator=accelerator,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback,early_stopping,lr_monitor],
                         gradient_clip_val=args.gradient_clip_val,
                         precision='16-mixed')
    trainer.fit(module,train_loader,val_loader)
    print(f'Best model: {checkpoint_callback.best_model_path}')
