import torch
from models.optimus_5prime.model import MeanRibosomeLoadModule
from models.optimus_5prime.data import *
from seqopt.oracle import NucleotideDesigner
from trials import parse_args, setup_model_from_lightning, lower_confidence_bound,run_all_trials,tune_langevin
from functools import partial

def evaluate(dataloader,model,test_df):

    storage = []
    with torch.no_grad():
        for batch in dataloader:
            utr,seq_id = batch['utr'],batch['seq_id']
            utr = torch.stack(batch['utr'],dim=0).to(device)
            utr = torch.nn.functional.one_hot(utr,num_classes=4).float()
            y_pred = model(utr)
            entry = {'seq_id' : seq_id[0],'mrl' : y_pred.item()}
            storage.append(entry)
    results = pd.DataFrame(storage)
    results['extrapolated'] = results['mrl'] > 1.164601
    results['trial'] = [t.split('_')[-1] for t in results['seq_id']]
    for trial, sub_df in results.groupby('trial'):
        count = sum([1 for x in sub_df['mrl'] if x > 1.164601])
        print(f'{trial}: {count}/{len(sub_df)} = {count/len(sub_df):.3f}')
        #print(sub_df['mrl'].describe())

if __name__ == "__main__":

    args = parse_args()
    device = "cpu" if args.device < 0 else f"cuda:{args.device}"
    module = MeanRibosomeLoadModule.load_from_checkpoint(args.checkpoint,
                                                         map_location=device)
    
    # extact PyTorch model and wrap 
    model = module.model
    model.eval()

    # test designed seq 
    test_df = pd.read_csv('short_runs.csv')
    test_df = test_df[~test_df['trial'].str.startswith('MCMC')]
    test_df.rename({'optimized_seq' : 'utr'},axis=1,inplace=True)
    dataset = PolysomeEGFP(test_df)
    dataloader = dataloader_from_dataset(dataset,1)
    evaluate(dataloader,model,test_df)
    
    # re do with the original sequences
    test_df = pd.read_csv('short_runs.csv')
    test_df = test_df[~test_df['trial'].str.startswith('MCMC')]
    test_df.rename({'original_seq' : 'utr'},axis=1,inplace=True)
    dataset = PolysomeEGFP(test_df)
    dataloader = dataloader_from_dataset(dataset,1)
    evaluate(dataloader,model,test_df)
