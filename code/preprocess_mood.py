from os import terminal_size
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

    # Gives each set of lyrics its own list
    # clean the data
    stop_words = set(stopwords.words('english'))

    lyrics = [[word.lower().strip("!()-',.?*{};:¡\"“‘~…’—–”\\")
               for word in song.split() if word not in stop_words] for song in lyrics]

    mean = np.mean([len(song) for song in lyrics])
    std = np.std([len(song) for song in lyrics])

    upper_bound = mean + 2*std
    lower_bound = mean - 2*std

    # helps with length of song/filtering out bad data
    indices = np.nonzero([1 if len(song) <= upper_bound and len(
        song) >= lower_bound else 0 for song in lyrics])[0]

    lyrics = [lyrics[i] for i in indices]
    labels = [labels[i] for i in indices]

    # shortens/evens out the list of lyrics
    for song in range(len(lyrics)):
        lyrics[song] = lyrics[song][:50]

    # assigns each label a number --> one hot encodes labels
    indices = [0 if x == 'Tension' else 1 if x ==
               'Tenderness' else 2 for x in labels]

    labels = tf.one_hot(indices, 3, dtype=tf.int64)

    # If we want one list of all lyrics:
    # train_lyrics_list = []
    # for x in train_lyrics:
    #     train_lyrics_list.append(x.split())
    # test_lyrics_list = []
    # for y in test_lyrics:
    #     test_lyrics_list.append(y.split())

    # gets rid of duplicate data
    unique = []
    for song in lyrics:
        unique.extend(song)
    unique = sorted(set(unique))

    vocabulary = {w: i for i, w in enumerate(unique, start=1)}

    lyrics = [list(map(lambda x: vocabulary[x], song))
              for song in lyrics]

    # pads data so it can be converted to tensor eventually and for even training
    lyrics = tf.keras.preprocessing.sequence.pad_sequences(
        lyrics, padding='post')  # returns np array

    # singlelabel (math):
    # total = 1103, 80% = 882, 20% = 221
    # labeled_lyrics
    # total = 150568, 80% = 120,454, 20% = 30,114

    # batch data
    index_range = tf.random.shuffle(range(len(lyrics)))
    shuffled_lyrics = tf.gather(lyrics, index_range)
    shuffled_labels = tf.gather(labels, index_range)

    train_lyrics, test_lyrics = shuffled_lyrics[:882], shuffled_lyrics[882:]
    train_labels, test_labels = shuffled_labels[:882], shuffled_labels[882:]

    # convert everything to a tensor and return data
    return tf.convert_to_tensor(train_lyrics), tf.convert_to_tensor(test_lyrics), train_labels, test_labels


def main():
    # just for testing
    # add print statements as necessary to test for proper output

    X0, Y0, X1, Y1 = get_data(
        "data/singlelabel.csv")

    return


if __name__ == '__main__':
    main()
