import tensorflow as tf
import glob
import json
import os
import sys
import math
from natsort import natsorted
import numpy as np

# adapted from https://github.com/phil-fradkin/contrastive_rna/blob/main/contrastive_rna_representation/saluki_dataset.py

def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type="ZLIB")

def parse_proto(example_protos,length_t=12288):
    """Parse TFRecord protobuf."""

    feature_spec = {
        "lengths": tf.io.FixedLenFeature((1,), tf.int64),
        "sequence": tf.io.FixedLenFeature([], tf.string),
        "coding": tf.io.FixedLenFeature([], tf.string),
        "splice": tf.io.FixedLenFeature([], tf.string),
        "targets": tf.io.FixedLenFeature([], tf.string)}

    # parse example into features
    feature_tensors = tf.io.parse_single_example(
        example_protos, features=feature_spec
    )
    # decode targets
    targets = tf.io.decode_raw(feature_tensors["targets"], tf.float16)
    targets = tf.cast(targets, tf.float32)
    
    # decode sequence
    sequence = tf.io.decode_raw(feature_tensors["sequence"], tf.uint8)

    # decode coding frame
    coding = tf.io.decode_raw(feature_tensors["coding"], tf.uint8)
    coding = tf.expand_dims(coding, axis=1)
    coding = tf.cast(coding, tf.float32)

    # decode splice
    splice = tf.io.decode_raw(feature_tensors["splice"], tf.uint8)
    splice = tf.expand_dims(splice, axis=1)
    splice = tf.cast(splice, tf.float32)

    # pad to zeros to full length
    # get length
    # seq_lengths = feature_tensors["lengths"]
    #paddings = [[0, length_t - seq_lengths[0]], [0, 0]]
    #inputs = tf.pad(inputs, paddings)
    return sequence,splice,coding,targets

def tfr_to_numpy(data_dir,split_label):
        """Convert TFR inputs and/or outputs to numpy arrays."""
        
        vocab = {0: "A", 1: "C", 2: "G", 3: "T", 4: "<reg>"} 
        with tf.name_scope("numpy"):
            # initialize dataset from TFRecords glob
            tfr_path = "%s/tfrecords/%s-*.tfr" % (data_dir, split_label)
            tfr_files = natsorted(glob.glob(tfr_path))
            if tfr_files:
                dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
            else:
                print("Cannot order TFRecords %s" % tfr_path, file=sys.stderr)
                dataset = tf.data.Dataset.list_files(tfr_path)

            # read TF Records
            dataset = dataset.flat_map(file_to_records)
            dataset = dataset.map(parse_proto)
            dataset = dataset.batch(1)

            storage = []
            # collect inputs and outputs
            for i,(seq_1hot, splice,coding, targets1) in enumerate(dataset):
                # sequence
                seq = seq_1hot.numpy().squeeze(0)
                seq = ''.join([vocab[x] for x in seq.tolist()])
                tgt = targets1.numpy().squeeze(0)[0]
                splice = splice.numpy().squeeze().tolist()
                coding = coding.numpy().squeeze().tolist()
                entry = {'id' : f'id_{i}',
                         'seq' : seq,
                         'seq_len' : len(seq),
                         'splice_sites' : splice,
                         'codon_starts' : coding, 
                         'half_life' : f'{tgt:.6f}'}
                storage.append(entry)

        # write to json
        with open(f'{data_dir}/{split_label}.json','w') as f:
            for entry in storage:
                f.write(json.dumps(entry)+'\n')

if __name__ == "__main__":

    data_dir = "data/saluki_agarwal_kelley/f0_c0/data0/"
    tfr_to_numpy(data_dir,'train')
    tfr_to_numpy(data_dir,'valid')
    tfr_to_numpy(data_dir,'test')