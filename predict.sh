#export BIOHOME=/nfs/stak/users/valejose/hpc-share
export BIOHOME=/home/bb/valejose/valejose/
export PYTHONPATH=$BIOHOME/bioseq2seq

python $PYTHONPATH/bioseq2seq/bin/translate.py --checkpoint seq_CNN.pt --input ${1} --mode bioseq2seq \
            --output_name designed_seq_LFN --num_gpus 1 --beam_size 4 --n_best 4 --max_tokens 1200 --max_decode_len 400 

