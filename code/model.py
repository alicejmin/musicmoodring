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
        self.embedding_size = ?
        self.vocab_size = None # pick a number 

        # need embedding layer??

        # look at paper for proper sizes and hypers

        # 1D conv layer
        # how should we determine hypers here
        self.conv1d = tf.keras.layers.Conv1D(1, 3)
        # max pool
        self.max_pool = tf.keras.layers.MaxPool1D() #or tf.nn.max_pool()
        # LSTM
        self.LSTM = tf.keras.layers.LSTM(self.embedding_size)
        # Dropout
        self.dropout = tf.keras.layers.Dropout(.5)
        # dense
        self.dense1 = tf.keras.layers.Dense()  # what activation
        # dropout
        self.dropout2 = tf.keras.layers.Dropout(.5)
        # dense
        self.dense2 = tf.keras.layers.Dense()  # what activation

    def call(self, inputs):

        # apply layers to inputs and return logits
        logits = self.conv1d(inputs)
        logits = self.max_pool(logits)
        logits = self.LSTM(logits)
        logits = self.dropout(logits)
        logits = self.dense1(logits)
        logits = self.dropout2(logits)
        logits = self.dense2(logits)

        return logits

    def loss(self, logits, labels):

        # define a reasonable loss function
        # (in the paper they use valance and arousal? maybe we pick something else)

        pass

    def accuracy(self, logits, labels):

        # define a reasonable accuracy function
        # (maybe different from paper)
        pass


def train(model, train_inputs, train_labels):

    # POSSIBLE WE DO NOT NEED THIS -- CAN WE JUST USE .fit AND .evalutate IN MAIN?
    
    #lucy i feel like it would be more accurate/better if we used gradient tape?
    #or do .fit and .evaluate cover that for you

    # train model -- maybe shuffle inputs -- look at hw2 and hw3
    # return average accuracy? maybe loss too (looking at one epoc only)
    pass


def test(model, test_inputs, test_labels):

    # POSSIBLE WE DO NOT NEED THIS -- CAN WE JUST USE .fit AND .evalutate IN MAIN?

    #same question as above

    # Tests the model on the test inputs and labels.
    # return average accuracy?
    pass


def main():
    # read in data using get_data()
    # initilize model
    # train and test model for set number of epochs

    return


if __name__ == '__main__':
    main()
