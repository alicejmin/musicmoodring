import nltk
from audioop import avg
import tensorflow as tf
import numpy as np
import pickle
import csv
import pandas as pd
from functools import reduce
from nltk.corpus import stopwords


def get_data(file_path):
    data = pd.read_csv(file_path)

    lyrics = data['lyrics']
    labels = data['label']

    # IF WE WANT EACH SET OF LYRICS TO HAVE ITS OWN LIST

    stop_words = set(stopwords.words('english'))

    lyrics = [[word.lower().strip("!()-',.?*{};:¡\"“‘~…’—–”\\")
               for word in song.split() if word not in stop_words] for song in lyrics]

    mean = np.mean([len(song) for song in lyrics])
    std = np.std([len(song) for song in lyrics])

    upper_bound = mean + 2*std
    lower_bound = mean - 2*std

    indices = np.nonzero([1 if len(song) <= upper_bound and len(
        song) >= lower_bound else 0 for song in lyrics])[0]

    lyrics = [lyrics[i] for i in indices]
    labels = [labels[i] for i in indices]

    for song in range(len(lyrics)):
        lyrics[song] = lyrics[song][:50]

    # indices = [0 if x == 'Sadness' else 1 if x ==
            #    'Tension' else 2 for x in labels]

    # labels = tf.one_hot(indices, 3, dtype=tf.int64)

    # IF WE WANT ONE LIST FOR ALL LYRICS
    # train_lyrics_list = []
    # for x in train_lyrics:
    #     train_lyrics_list.append(x.split())
    # test_lyrics_list = []
    # for y in test_lyrics:
    #     test_lyrics_list.append(y.split())

    unique = []
    for song in lyrics:
        unique.extend(song)
    unique = sorted(set(unique))

    vocabulary = {w: i for i, w in enumerate(unique, start=1)}

    lyrics = [list(map(lambda x: vocabulary[x], song))
              for song in lyrics]

    lyrics = tf.keras.preprocessing.sequence.pad_sequences(
        lyrics, padding='post')  # returns np array
    #singlelabel:
    # total = 1103, 80% = 882, 20% = 221
    #labeled_lyrics
    # total = 150568, 80% = 120,454, 20% = 30,114
    print(len(lyrics))
    train_lyrics, test_lyrics = lyrics[:120454], lyrics[120454:]
    train_labels, test_labels = labels[:120454], labels[120454:]

    return tf.convert_to_tensor(train_lyrics), tf.convert_to_tensor(test_lyrics), tf.convert_to_tensor(train_labels), tf.convert_to_tensor(test_labels)


def main():
    # can delete later -- just for testing

    X0, Y0, X1, Y1 = get_data(
        "data/labeled_lyrics_cleaned.csv")

    print(X0)
    # print(Y0)
    print(X1)
    # print(Y1)

    return


if __name__ == '__main__':
    main()
