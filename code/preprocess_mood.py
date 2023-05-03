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

    # lyrics_sad = [lyrics[i] for i in range(len(labels)) if labels[i] == "Sadness"]
    # labels_sad = [labels[i] for i in range(len(labels)) if labels[i] == "Sadness"]

    # lyrics_tender = [lyrics[i] for i in range(len(labels)) if labels[i] == "Tenderness"]
    # labels_tender = [labels[i] for i in range(len(labels)) if labels[i] == "Tenderness"]

    # lyrics_tension = [lyrics[i] for i in range(len(labels)) if labels[i] == "Tension"]
    # labels_tension = [labels[i] for i in range(len(labels)) if labels[i] == "Tension"]

    # lyrics_sad = lyrics_sad[:267]
    # labels_sad = labels_sad[:267]
    # # print(len(lyrics_sad))

    # lyrics_tender = lyrics_tender[:267]
    # labels_tender = labels_tender[:267]
    # # print(len(lyrics_tender))

    # # print(len(lyrics_tension))
    # lyrics = np.concatenate((lyrics_sad, lyrics_tender, lyrics_tension))
    # labels = np.concatenate((labels_sad, labels_tender, labels_tension))
    # print(len(lyrics))
    for song in range(len(lyrics)):
        lyrics[song] = lyrics[song][:50]
    
    indices = [0 if x == 'Tension' else 1 if x ==
               'Tenderness' else 2 for x in labels]
    
    # tot = list(zip(lyrics, labels))
    # sad = 0
    # tender = 0 
    # tension = 0
    # for i in tot:
    #     if i[1] == 0 and sad > 300:
    #         sad +=1
    #         tot.remove(i)
    #     elif i[1] ==1 and tension > 300:
    #         tension +=1
    #         tot.remove(i)
    #     elif i[1] == 2 and tender > 200: 
    #         tender +=1
    #         tot.remove(i)
    #     elif i[1] == 0:
    #         sad +=1
    #     elif i[1] ==1:
    #         tension +=1
    #     else: 
    #         tender +=1
    # lyrics, indices = zip(*tot)

    labels = tf.one_hot(indices, 3, dtype=tf.int64)

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
    # singlelabel:
    # total = 1103, 80% = 882, 20% = 221
    # labeled_lyrics
    # total = 150568, 80% = 120,454, 20% = 30,114

    # 827

    index_range = tf.random.shuffle(range(len(lyrics)))
    shuffled_lyrics = tf.gather(lyrics, index_range)
    shuffled_labels = tf.gather(labels, index_range)

    train_lyrics, test_lyrics = shuffled_lyrics[:882], shuffled_lyrics[882:]
    train_labels, test_labels = shuffled_labels[:882], shuffled_labels[882:]
    # print(train_labels)
    # print(test_labels)

    return tf.convert_to_tensor(train_lyrics), tf.convert_to_tensor(test_lyrics), train_labels, test_labels


def main():
    # can delete later -- just for testing

    X0, Y0, X1, Y1 = get_data(
        "data/singlelabel.csv")

    #print(X0)
    # print(Y0)
    #print(X1)
    # print(Y1)

    return


if __name__ == '__main__':
    main()
