import tensorflow as tf
import numpy as np
import pickle
import csv
import pandas as pd 


def get_data(file_path):
    # read in data
    data = pd.read_csv(file_path)

    lyrics = data['lyrics']  # or maybe change to index?
    labels = data['label']  # or maybe change to index?
    # split into labels and lyrics

    # total = 1160, 80% = 928, 20% = 232

    labels = tf.one_hot(labels, 9)  # not sure if this is right -- 9 classes?

    train_lyrics, test_lyrics = lyrics[:928], lyrics[929:]
    train_labels, test_labels = labels[:928], labels[929:]

    # already in 1-D arrays so don't need to flatten

    # need to reshape?? normalize? one hot encode?

    return tf.convert_to_tensor(train_lyrics), tf.convert_to_tensor(test_lyrics), tf.convert_to_tensor(train_labels), tf.convert_to_tensor(test_labels)
