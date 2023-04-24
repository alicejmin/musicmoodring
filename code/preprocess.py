import tensorflow as tf
import numpy as np
import pickle
import csv
import pandas as pd
from functools import reduce


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

    # total = 1160, 80% = 928, 20% = 232
    train_lyrics, test_lyrics = lyrics[:928], lyrics[928:]
    train_labels, test_labels = labels[:928], labels[928:]

    # IF WE WANT EACH SET OF LYRICS TO HAVE ITS OWN LIST

    train_lyrics = [[word.lower().strip("!()-',.?*{};:¡\"“‘~…’—–”")
                     for word in song.split()] for song in train_lyrics]
    test_lyrics = [[word.lower().strip("!()-',.?*{};:¡\"“‘~…’—–”")
                    for word in song.split()] for song in test_lyrics]

    # IF WE WANT ONE LIST FOR ALL LYRICS
    # train_lyrics_list = []
    # for x in train_lyrics:
    #     train_lyrics_list.append(x.split())
    # test_lyrics_list = []
    # for y in test_lyrics:
    #     test_lyrics_list.append(y.split())

    train_unique = [sorted(set(x)) for x in train_lyrics]
    test_unique = [sorted(set(x)) for x in test_lyrics]
    unique = []
    for song in train_unique:
        unique_line = [word for word in song if word not in unique]
        unique.extend(unique_line)
    for song in test_unique:
        unique_line = [word for word in song if word not in unique]
        unique.extend(unique_line)
    unique = sorted(unique)

    # do we need to limit it to vocab_size?

    vocabulary = {w: i for i, w in enumerate(unique)}

    train_lyrics = [list(map(lambda x: vocabulary[x], song))
                    for song in train_lyrics]
    test_lyrics = [list(map(lambda x: vocabulary[x], song))
                   for song in test_lyrics]

    # already in 1-D arrays so don't need to flatten? need to reshape?? normalize?

    # error here with converting to tensor - try ragged tensor? or somehow even out lyric lengths?
    return tf.convert_to_tensor(train_lyrics), tf.convert_to_tensor(test_lyrics), tf.convert_to_tensor(train_labels), tf.convert_to_tensor(test_labels)


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
