# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:50:19 2024

@author: manodeep ray
"""
from load_data import load_dataset , train_val_label
from tools import squeeze , get_spectrogram , make_spec_ds , prefetch_cache_shuffle , run_inference
import numpy as np
import tensorflow as tf
from plot_functions import plot_example , set_axes , plot_spectrogram , display_audio_label , plot_model_metrics , plot_confusion_matrix
from IPython import display
from models import load_model , comlpile_model , train_model
import matplotlib.pyplot as plt
def main():
    
    
    
    seed = 42   
    tf.random.set_seed(seed)
    np.random.seed(seed)
    print("loaing dataset...")

    
    DATASET_PATH = 'data/mini_speech_commands'
    commands , data_dir= load_dataset(DATASET_PATH)
    train_data, val_data , label_names = train_val_label(DATASET_PATH)
    
    print(train_data.element_spec)
    print(val_data.element_spec)
    print("squeezing audio files...")
    
    train_ds = train_data.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_data.map(squeeze, tf.data.AUTOTUNE)
    
    print("sharding data...")

    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)
    
    print("plotting waveform of few audio files...")

    example_audio, example_labels = plot_example(train_ds,label_names)
    
    spectrogram , label , waveform = display_audio_label(example_audio, example_labels,label_names)
    
    print("plotting spectogram of few audio files...")

    
    axes = set_axes(waveform)
    plot_spectrogram(spectrogram, axes[1])
    axes[1].set_title('Spectrogram')
    plt.suptitle(label.title())
    plt.show()
    
    
    print("storing spectograms of  audio files into train,  test and val datasets ...")

    train_spectrogram_ds = make_spec_ds(train_ds)
    val_spectrogram_ds = make_spec_ds(val_ds)
    test_spectrogram_ds = make_spec_ds(test_ds)

    for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
        break
    
    rows = 3
    cols = 3
    n = rows*cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
    print("plotting spectogram of few audio files...")

    for i in range(n):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        plot_spectrogram(example_spectrograms[i].numpy(), ax)
        ax.set_title(label_names[example_spect_labels[i].numpy()])
    
    plt.show()    
    
    train_spectrogram_ds,val_spectrogram_ds,test_spectrogram_ds = prefetch_cache_shuffle(train_spectrogram_ds,val_spectrogram_ds,test_spectrogram_ds)
    print(" loading model...")

    model = load_model(label_names ,example_spectrograms,train_spectrogram_ds)
    print("finished loading model...")
    
    print(" compiling model...")
    
    model = comlpile_model(model)
    print("finished compiling model...")
    
    EPOCHS = 10
    
    print(" training model...")
    history = train_model(model,val_spectrogram_ds,train_spectrogram_ds , EPOCHS)
    
    print("finished training model...")
    
    print("plotting metrics...")
    plot_model_metrics(history)
    
    print("model evaluaton : ")
    model.evaluate(test_spectrogram_ds, return_dict=True)
    
    print("plotting confusion matrix...")
    plot_confusion_matrix(model, test_spectrogram_ds,label_names)
    
    x = data_dir/'no/01bb6a2a_nohash_0.wav'
    
    
    
    print(" running inference...")
    run_inference(model, x)
    print("finished running inference...")
    
    print("finished ...")


if __name__ == "__main__":
    main()

    