# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:47:29 2024

@author: manodeep
"""

import tensorflow as tf
import os
import numpy as np
import pathlib


def load_dataset(DATASET_PATH):
    data_dir = pathlib.Path(DATASET_PATH)
    if not data_dir.exists():
            tf.keras.utils.get_file(
              'mini_speech_commands.zip',
              origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
              extract=True,
              cache_dir='.', cache_subdir='data')
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
    print('Commands:', commands)
    return commands , data_dir

def train_val_label(DATASET_PATH):
    data_dir = pathlib.Path(DATASET_PATH)

    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000,
    subset='both')
    
    label_names = np.array(train_ds.class_names)
    print()
    print("label names:", label_names)
    
    return train_ds, val_ds , label_names
