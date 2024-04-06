# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:59:44 2024

@author: manodeep
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

def get_spectrogram(waveform):
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def make_spec_ds(ds):
    return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

def prefetch_cache_shuffle(  train_spectrogram_ds,val_spectrogram_ds,test_spectrogram_ds):
    train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
    return  train_spectrogram_ds,val_spectrogram_ds,test_spectrogram_ds



def run_inference(model , x):
    x = tf.io.read_file(str(x))
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
    x = tf.squeeze(x, axis=-1)
    waveform = x
    x = get_spectrogram(x)
    x = x[tf.newaxis,...]
    
    prediction = model(x)
    x_labels = ['no', 'yes', 'down', 'go', 'left', 'up', 'right', 'stop']
    plt.bar(x_labels, tf.nn.softmax(prediction[0]))
    plt.title('No')
    plt.show()
    
    display.display(display.Audio(waveform, rate=16000))
        