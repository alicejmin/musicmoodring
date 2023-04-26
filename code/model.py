from preprocess import get_data
import tensorflow as tf
import numpy as np


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        # initialize layers and hyperparameters
        # batch size
        # number of classes (moods)
        # learning rate
        # epochs
        # stride
        # optimizer
        # padding
        self.batch_size = 32  # no idea
        self.num_classes = 9
        self.lr = .01
        self.epochs = 10
        # self.stride = (default is 1 so only need this if want something different?)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr)
        self.padding = "SAME"
        self.embedding_size = 64 #need to change later
        self.vocab_size = 128 # pick a number 
        self.hidden_size = 256 #need to change later
        

        # need embedding layer??
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)

        # look at paper for proper sizes and hypers

        # 1D conv layer
        # how should we determine hypers here
        self.conv1d = tf.keras.layers.Conv1D(1, 3, strides=1, padding=self.padding)
        # max pool
        # self.max_pool = tf.keras.layers.MaxPool1D() #or 
        self.max_pool = tf.nn.max_pool(self.conv1d, [3, 3], strides=1, padding=self.padding) # input?? change -- should not be self.conv1d
        # LSTM
        self.LSTM = tf.keras.layers.LSTM(self.embedding_size, activation="leaky_relu")
        # Dropout
        self.dropout = tf.keras.layers.Dropout(.5)
        # dense
        self.dense1 = tf.keras.layers.Dense(self.hidden_size, activation="leaky_relu")  # what activation
        # dropout
        self.dropout2 = tf.keras.layers.Dropout(.5)
        # dense
        self.dense2 = tf.keras.layers.Dense(self.vocab_size, activation="softmax")  # what activation
        #output of last dense layer should be vocab size, activation should be softmax

        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def call(self, inputs):

        # apply layers to inputs and return logits
        logits = self.embedding(inputs)
        logits = self.conv1d(logits)
        logits = self.max_pool(logits)
        logits = self.LSTM(logits)
        logits = self.dropout(logits)
        logits = self.dense1(logits)
        logits = self.dropout2(logits)
        logits = self.dense2(logits)

        return logits

    def accuracy(self, logits, labels):

        # define a reasonable accuracy function
        # (maybe different from paper)
        cross_ent = tf.math.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(labels, logits)) #.losses or .metrics
        perplex = tf.math.exp(cross_ent)
        return perplex



def train(model, train_lyrics, train_labels):

    # POSSIBLE WE DO NOT NEED THIS -- CAN WE JUST USE .fit AND .evalutate IN MAIN?
    
    #lucy i feel like it would be more accurate/better if we used gradient tape?
    #or do .fit and .evaluate cover that for you

    # train model -- maybe shuffle inputs -- look at hw2 and hw3
    # return average accuracy? maybe loss too (looking at one epoc only)
    avg_acc = 0
    counter = 0
    for batch_num, b1 in enumerate(range(model.batch_size, train_lyrics.shape[0] + 1, model.batch_size)):
        b0 = b1 - model.batch_size
        batch_lyrics = train_lyrics[b0:b1]
        batch_labels = train_labels[b0:b1]

        with tf.GradientTape() as tape:
                logits = model(batch_lyrics) 
                loss = model.loss(batch_labels, logits)
        
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        acc = model.accuracy(logits, batch_labels)
        avg_acc += acc
        counter += 1
        print("TRAIN", "batch:", batch_num, "acc:", acc)
        print("TRAIN", "batch:", batch_num, "loss:", loss)
    
    print("average accuracy:", avg_acc/counter)

    return


def test(model, test_lyrics, test_labels):

    # POSSIBLE WE DO NOT NEED THIS -- CAN WE JUST USE .fit AND .evalutate IN MAIN?

    #same question as above

    # Tests the model on the test inputs and labels.
    # return average accuracy?

    avg_acc = 0
    counter = 0
    for batch_num, b1 in enumerate(range(model.batch_size, test_lyrics.shape[0] + 1, model.batch_size)):
        b0 = b1 - model.batch_size
        batch_lyrics = test_lyrics[b0:b1]
        batch_labels = test_labels[b0:b1]

        logits = model(batch_lyrics)
        loss = model.loss(batch_labels, logits)

        acc = model.accuracy(logits, batch_labels)
        avg_acc += acc
        counter += 1
        print("TEST", "batch:", batch_num, "acc:", acc)
        print("TEST", "batch:", batch_num, "loss:", loss)

    return avg_acc/counter


def main():
    # read in data using get_data()
    # initilize model
    # train and test model for set number of epochs

    train_lyrics, test_lyrics, train_labels, test_labels = get_data(
        "data/singlelabel.csv")
    
    model = Model()

    for _ in range(model.epochs):
        train(model, train_lyrics, train_labels)

    t = test(model, test_lyrics, test_labels)
    
    print("FINAL TEST ACC:", t)

    return


if __name__ == '__main__':
    main()
