# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:32:20 2024

@author: manodeep
"""
import tensorflow as tf 
from tools import get_spectrogram
class ExportModel(tf.Module):
  def __init__(self, model,label_names):
    self.model = model

    # Accept either a string-filename or a batch of waveforms.
    # YOu could add additional signatures for a single wave, or a ragged-batch. 
    self.__call__.get_concrete_function(
        x=tf.TensorSpec(shape=(), dtype=tf.string))
    self.__call__.get_concrete_function(
       x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))
    self.label_names = label_names

  @tf.function
  def __call__(self, x,label_names):
    # If they pass a string, load the file and decode it. 
    if x.dtype == tf.string:
      x = tf.io.read_file(x)
      x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
      x = tf.squeeze(x, axis=-1)
      x = x[tf.newaxis, :]

    x = get_spectrogram(x)  
    result = self.model(x, training=False)

    class_ids = tf.argmax(result, axis=-1)
    class_names = tf.gather(self.label_names, class_ids)
    return {'predictions':result,
            'class_ids': class_ids,
            'class_names': class_names}


