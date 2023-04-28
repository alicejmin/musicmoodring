from preprocess import get_data
import tensorflow as tf
import numpy as np


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.batch_size = 32 
        self.num_classes = 3
        self.lr = .01
        self.epochs = 10
        # self.stride = (default is 1 so only need this if want something different?)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr)
        self.padding = "SAME"
        self.embedding_size = 100 
        self.vocab_size = 15245 
        self.hidden_size = 256 
        
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, embeddings_initializer="zero") #the default initializer here is "uniform" we can play around with it

        self.permute = tf.keras.layers.Permute((2, 1), input_shape=(529, 64))

        #did some research... top three seem to be HeNormal, Kaiming, and Xavier but dont know which is best
        self.conv1d = tf.keras.layers.Conv1D(16, 2, strides=2, padding=self.padding, activation="relu", kernel_initializer="HeNormal") #I think HeNormal(kaiming) is best, top  seem to be xavier, and he normal
        
        # flatten?
        self.permute2 = tf.keras.layers.Permute((2, 0, 1), input_shape=(529, 64))
        # LSTM
        self.LSTM = tf.keras.layers.LSTM(self.embedding_size, activation="leaky_relu") # what size??
        # Dropout
        self.dropout = tf.keras.layers.Dropout(.5)
        # dense
        self.dense1 = tf.keras.layers.Dense(self.hidden_size, activation="tanh") # paper output is 64
        # dropout
        self.dropout2 = tf.keras.layers.Dropout(.5)
        # dense
        self.dense2 = tf.keras.layers.Dense(self.num_classes, activation="softmax") # keep softmax??

        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.loss = tf.keras.losses.CategoricalCrossentropy()

    def call(self, inputs):

        logits = self.embedding(inputs) 
        print("2:", logits.shape) # [32, 529, 100]
        logits = self.permute(logits) 
        print("2.5:", logits.shape) # [32, 100, 529]
        logits = self.conv1d(logits)
        logits = tf.nn.max_pool(logits, 2, strides=None, padding=self.padding) # 3 or [3, 3]?
        print("4:", logits.shape) # [32, 50, 16]
        logits = self.permute2(logits)
        print("4:", logits.shape)
        logits = self.LSTM(logits)
        print("5:", logits.shape) # [32, 100]
        logits = self.dropout(logits)
        print("6:", logits.shape) # [32, 100]
        logits = self.dense1(logits)
        print("7:", logits.shape) # [32, 256]
        logits = self.dropout2(logits)
        print("8:", logits.shape) # [32, 256]
        logits = self.dense2(logits)
        print("9:", logits.shape) # [32, 3]

        return logits

    def accuracy(self, logits, labels):

        # cross_ent = tf.math.reduce_mean(self.loss(labels, logits)) #.losses or .metrics
        # perplex = tf.math.exp(cross_ent)
        # return perplex

        num_correct_classes = 0
        for song in range(logits.shape[0]):
            if tf.argmax(logits[song], axis=-1) == tf.argmax(labels[song]):
                num_correct_classes += 1 
        accuracy = num_correct_classes/logits.shape[0]
        return accuracy



def train(model, train_lyrics, train_labels):

    # train model -- maybe shuffle inputs -- look at hw2 and hw3

    avg_acc = 0
    avg_loss = 0
    counter = 0
    for batch_num, b1 in enumerate(range(model.batch_size, len(train_lyrics) + 1, model.batch_size)): # train_lyrics.shape[0] + 1 if tensor
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
        avg_loss += loss
        counter += 1

        print(f"\r[Train {batch_num+1}/{27}]\t loss={loss:.3f}\t acc: {acc:.3f}", end='')

    print()
    return avg_loss/counter, avg_acc/counter


def test(model, test_lyrics, test_labels):

    # Tests the model on the test inputs and labels.

    avg_acc = 0
    avg_loss = 0
    counter = 0
    for batch_num, b1 in enumerate(range(model.batch_size, len(test_lyrics) + 1, model.batch_size)): # make test_lyrics.shape[0] + 1?
        b0 = b1 - model.batch_size
        batch_lyrics = test_lyrics[b0:b1]
        batch_labels = test_labels[b0:b1]

        logits = model(batch_lyrics)
        loss = model.loss(batch_labels, logits)

        acc = model.accuracy(logits, batch_labels)
        avg_acc += acc
        avg_loss += loss
        counter += 1
        print(f"\r[Valid {batch_num+1}/{6}]\t loss={loss:.3f}\t acc: {acc:.3f}", end='')

    print()
    return avg_acc/counter, avg_loss/counter


def main():

    train_lyrics, test_lyrics, train_labels, test_labels = get_data(
        "data/singlelabel.csv")
    
    model = Model()

    for e in range(model.epochs):
        print("epoch", e+1)
        train(model, train_lyrics, train_labels)

    t = test(model, test_lyrics, test_labels)

    tf.print("Final Accuracy:", t[0])

    return


if __name__ == '__main__':
    main()
