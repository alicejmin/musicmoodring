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
        self.num_classes = 3
        self.lr = .01
        self.epochs = 10
        # self.stride = (default is 1 so only need this if want something different?)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr)
        self.padding = "SAME"
        self.embedding_size = 64 #need to change later
        self.vocab_size = 15245 # change later
        self.hidden_size = 256 #need to change later
        

        # need embedding layer??
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size,embeddings_initializer="zero") #the default initializer here is "uniform" we can play around with it

        # look at paper for proper sizes and hypers

        #did some research... top three seem to be HeNormal, Kaiming, and Xavier but dont know which is best still getting nan
        # 1D conv layer
        # how should we determine hypers here
        self.conv1d = tf.keras.layers.Conv1D(1, 3, strides=1, padding=self.padding, kernel_initializer="HeNormal") #I think HeNormal(kaiming) is best, top  seem to be xavier, and he normal
        # max pool
        # self.max_pool = tf.keras.layers.MaxPool1D() #or 
        #self.max_pool = tf.nn.max_pool(self.conv1d, [3, 3], strides=1, padding=self.padding) # input?? change -- should not be self.conv1d
        # LSTM
        self.LSTM = tf.keras.layers.LSTM(self.embedding_size, activation="leaky_relu") #these also all have initializers
        # Dropout
        self.dropout = tf.keras.layers.Dropout(.5)
        # dense
        self.dense1 = tf.keras.layers.Dense(self.hidden_size, activation="leaky_relu")  # what activation
        # dropout
        self.dropout2 = tf.keras.layers.Dropout(.5)
        # dense
        self.dense2 = tf.keras.layers.Dense(self.num_classes, activation="softmax")  # what activation
        #output of last dense layer should be vocab size, activation should be softmax
        # change this last layer to be num_examples? 

        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.loss = tf.keras.losses.CategoricalCrossentropy() #tf.keras.losses.SparseCategoricalCrossentropy()

    def call(self, inputs):

        logits = self.embedding(inputs)

        logits = self.conv1d(logits)
        logits = tf.nn.max_pool(logits, 3, strides=1, padding=self.padding) # 3 or [3, 3]?
        logits = self.LSTM(logits)
        logits = self.dropout(logits)
        logits = self.dense1(logits)
        logits = self.dropout2(logits)
        logits = self.dense2(logits)

        return logits

    def accuracy(self, logits, labels):

        # define a reasonable accuracy function
        # (maybe different from paper)
        # hw 5 acc? 
        # correct_classes = tf.argmax(logits, axis=-1) == labels
        # accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(correct_classes, tf.float32), mask)) #not sure what mask input would be
        # return accuracy

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

    # POSSIBLE WE DO NOT NEED THIS -- CAN WE JUST USE .fit AND .evalutate IN MAIN?

    # train model -- maybe shuffle inputs -- look at hw2 and hw3
    # return average accuracy? maybe loss too (looking at one epoc only)

    # lyrics = (928, 529)
    # labels = (928, 3)
    avg_acc = 0
    avg_loss = 0
    counter = 0

    for batch_num, b1 in enumerate(range(model.batch_size, len(train_lyrics) + 1, model.batch_size)): # train_lyrics.shape[0] + 1 if tensor
        b0 = b1 - model.batch_size
        batch_lyrics = train_lyrics[b0:b1]
        batch_labels = train_labels[b0:b1]
        # batch_lyrics = (32, 529)
        # batch_labels = (32, 3)

        with tf.GradientTape() as tape:
            logits = model(batch_lyrics) 
            # logits = (32, 15244)
            # batch_labels = (32, 3)
            loss = model.loss(batch_labels, logits)
        
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        acc = model.accuracy(logits, batch_labels)
        
        avg_acc += acc
        avg_loss += loss
        counter += 1
        # print("TRAIN", "batch:", batch_num, "acc:", acc)
        # print("TRAIN", "batch:", batch_num, "loss:", loss)

        print(f"\r[Train {batch_num+1}/{model.batch_size}]\t loss={loss:.3f}\t acc: {acc:.3f}", end='')
        # print("average accuracy:", avg_acc/counter, "average loss:", avg_loss/counter)

    print()
    return avg_loss/counter, avg_acc/counter


def test(model, test_lyrics, test_labels):

    # POSSIBLE WE DO NOT NEED THIS -- CAN WE JUST USE .fit AND .evalutate IN MAIN?

    #same question as above

    # Tests the model on the test inputs and labels.
    # return average accuracy?

    avg_acc = 0
    avg_loss = 0
    counter = 0
    for batch_num, b1 in enumerate(range(model.batch_size, len(test_lyrics) + 1, model.batch_size)): # make test_lyrics.shape[0] + 1 if we get a tensor
        b0 = b1 - model.batch_size
        batch_lyrics = test_lyrics[b0:b1]
        batch_labels = test_labels[b0:b1]

        logits = model(batch_lyrics)
        loss = model.loss(batch_labels, logits)

        acc = model.accuracy(logits, batch_labels)
        avg_acc += acc
        avg_loss += loss
        counter += 1
        # print("TEST", "batch:", batch_num, "acc:", acc)
        # print("TEST", "batch:", batch_num, "loss:", loss)

    return avg_acc/counter, avg_loss/counter


def main():
    # read in data using get_data()
    # initilize model
    # train and test model for set number of epochs

    train_lyrics, test_lyrics, train_labels, test_labels = get_data(
        "data/singlelabel.csv")
    
    model = Model()

    for e in range(model.epochs):
        print("epoch", e+1)
        train(model, train_lyrics, train_labels)

    t = test(model, test_lyrics, test_labels)
    
    # print(f"\r[Valid {batch_num+1}/{counter}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}", end='')

    #made epochs four for now

    #print("FINAL TEST ACC:", t) #cant get this to print out of a tensor
    
    # print("FINAL", t)

    tf.print("Final Accuracy:", t[0]) #this is not printing the right thing/accuracy is really low for final for some reason
    return


if __name__ == '__main__':
    main()
