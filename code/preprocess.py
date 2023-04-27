from audioop import avg
import tensorflow as tf
import numpy as np
import pickle
import csv
import pandas as pd
from functools import reduce
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def get_data(file_path):
    # read in data
    data = pd.read_csv(file_path)

    # need to pickle/unpickle? purpose?

    # split into labels and lyrics
    lyrics = data['lyrics']
    labels = data['label']

    indices = [0 if x == 'Sadness' else 1 if x ==
               'Tension' else 2 for x in labels]

    labels = tf.one_hot(indices, 3)

    # IF WE WANT EACH SET OF LYRICS TO HAVE ITS OWN LIST

    stop_words = set(stopwords.words('english'))

    lyrics = [[word.lower().strip("!()-',.?*{};:¡\"“‘~…’—–”")
                     for word in song.split() if word not in stop_words] for song in lyrics]
    # test_lyrics = [[word.lower().strip("!()-',.?*{};:¡\"“‘~…’—–”")
    #                 for word in song.split() if word not in stop_words] for song in test_lyrics]

    # re?

    # IF WE WANT ONE LIST FOR ALL LYRICS
    # train_lyrics_list = []
    # for x in train_lyrics:
    #     train_lyrics_list.append(x.split())
    # test_lyrics_list = []
    # for y in test_lyrics:
    #     test_lyrics_list.append(y.split())

    unique = [sorted(set(x)) for x in lyrics]
    # test_unique = [sorted(set(x)) for x in test_lyrics]
    unique = []
    for song in unique:
        unique_line = [word for word in song if word not in unique]
        unique.extend(unique_line)
    # for song in test_unique:
    #     unique_line = [word for word in song if word not in unique]
    #     unique.extend(unique_line)
    unique = sorted(unique)

    vocabulary = {w: i for i, w in enumerate(unique)}

    lyrics = [list(map(lambda x: vocabulary[x], song))
                    for song in lyrics]
    # test_lyrics = [list(map(lambda x: vocabulary[x], song))
    #                for song in test_lyrics]
 
    # get rid of outliers 
    # find mean and std 
    mean = np.mean([len(song) for song in lyrics])
    # test_mean = np.mean([len(song) for song in test_lyrics])
    # mean = (train_mean + test_mean)/2
    std = np.std(lyrics)

    upper_bound = mean + 2*std
    lower_bound = mean - 2*std

    lyrics = [song for song in lyrics if len(song) <= upper_bound and len(song) >= lower_bound] 
    
    # tf.keras.preprocess.sequence.padsequence? padding post or pre? 

    train_lyrics = tf.keras.utils.pad_sequences(lyrics, value=-1, padding='post') # returns np array
    # test_lyrics = tf.keras.utils.pad_sequences(test_lyrics, value=-1, padding='post')

    # total = 1160, 80% = 928, 20% = 232
    train_lyrics, test_lyrics = lyrics[:928], lyrics[928:]
    train_labels, test_labels = labels[:928], labels[928:]

    # error here with converting to tensor - try ragged tensor? or somehow even out lyric lengths?
    # return tf.convert_to_tensor(train_lyrics), tf.convert_to_tensor(test_lyrics), tf.convert_to_tensor(train_labels), tf.convert_to_tensor(test_labels)
    return tf.convert_to_tensor(train_lyrics), tf.convert_to_tensor(test_lyrics), train_labels, test_labels
    # ask taishi about this (lyrics are lists, labels are tensors, having touble converting all to tensors so wondering if it is needed)


def main():
    # can delete later -- just for testing

    X0, Y0, X1, Y1 = get_data(
        "data/singlelabel.csv")

    print(X0)
    print(Y0)
    print(X1)
    print(Y1)

    return


if __name__ == '__main__':
    main()
