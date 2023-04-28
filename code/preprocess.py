import nltk
from audioop import avg
import tensorflow as tf
import numpy as np
import pickle
import csv
import pandas as pd
from functools import reduce
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def get_data(file_path):
    # read in data
    data = pd.read_csv(file_path)

    # need to pickle/unpickle? purpose?

    # split into labels and lyrics
    lyrics = data['lyrics']
    labels = data['label']

    # IF WE WANT EACH SET OF LYRICS TO HAVE ITS OWN LIST

    stop_words = set(stopwords.words('english'))

    lyrics = [[word.lower().strip("!()-',.?*{};:¡\"“‘~…’—–”\\")
                     for word in song.split() if word not in stop_words] for song in lyrics]
    

    mean = np.mean([len(song) for song in lyrics])
    # test_mean = np.mean([len(song) for song in test_lyrics])
    # mean = (train_mean + test_mean)/2
    std = np.std([len(song) for song in lyrics])

    upper_bound = mean + 2*std
    lower_bound = mean - 2*std

    indices = np.nonzero([1 if len(song) <= upper_bound and len(song) >= lower_bound else 0 for song in lyrics])[0]
    # lyrics = [song[np.nonzero(index)] for song in lyrics for index in indices]
    # labels = [song[np.nonzero(index)] for song in labels for index in indices]
    lyrics = np.take(lyrics, indices)
    labels = np.take(labels, indices)

    indices = [0 if x == 'Sadness' else 1 if x ==
               'Tension' else 2 for x in labels]

    labels = tf.one_hot(indices, 3, dtype=tf.int64)

    # [song for song in lyrics if len(song) <= upper_bound and len(song) >= lower_bound]

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

    # train_unique = sorted(set(train_data))
    # test_unique = sorted(set(test_data))
    # unique = sorted(set(train_unique + test_unique))

    # unique = [sorted(set(x)) for x in lyrics]
    # # test_unique = [sorted(set(x)) for x in test_lyrics]
    # unique = []
    # for song in unique:
    #     unique_line = [word for word in song if word not in unique]
    #     unique.extend(unique_line)
    # # for song in test_unique:
    # #     unique_line = [word for word in song if word not in unique]
    # #     unique.extend(unique_line)
    # unique = sorted(unique)

    unique = []
    for song in lyrics:
        unique.extend(song)
    unique = sorted(set(unique))

    vocabulary = {w: i for i, w in enumerate(unique, start=1)}

    lyrics = [list(map(lambda x: vocabulary[x], song))
                    for song in lyrics]
    # test_lyrics = [list(map(lambda x: vocabulary[x], song))
    #                for song in test_lyrics]
 
    # get rid of outliers 
    # find mean and std 
    
    # tf.keras.preprocess.sequence.padsequence? padding post or pre? 

    lyrics = tf.keras.preprocessing.sequence.pad_sequences(lyrics, padding='post') # returns np array
    print(len(lyrics[0]))
    # test_lyrics = tf.keras.utils.pad_sequences(test_lyrics, value=-1, padding='post')
    # total = 1103, 80% = 882, 20% = 221
    train_lyrics, test_lyrics = lyrics[:882], lyrics[882:] # fix bc nums won't be the same 
    train_labels, test_labels = labels[:882], labels[882:]

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